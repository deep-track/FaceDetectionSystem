#!/usr/bin/env bash
# DeepTrack API on ECS Fargate behind an ALB (account 097659826060, us-east-1).
# Safe to re-run: each step reuses what already exists.
#
#   bash infra/ecs/deploy.sh [image-tag]     # default: newest tag in ECR
#   bash infra/ecs/deploy.sh --https         # after the ACM cert is validated:
#                                            # add HTTPS and redirect HTTP to it
#
# Creates: security groups, an execution role that can read only the API key
# secret, a log group, the cluster, the task definition, an internet-facing
# ALB with a target group, the service (1 task), and an ACM certificate for
# api.deeptrack.io (DNS validation; the CNAME to add is printed).
# Uses the default VPC's public subnets, no NAT gateway: the task gets a
# public IP for pulling the image, and its security group only accepts the ALB.
set -euo pipefail

REGION=us-east-1
ACCOUNT=097659826060
NAME=deeptrack-api
DOMAIN=api.deeptrack.io
REPO=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$NAME
SECRET_ID=deeptrack-api/api-keys
export AWS_DEFAULT_REGION=$REGION
# Git Bash on Windows: aws.exe ends lines with CRLF, which would leave \r in
# every captured value.
if command -v cygpath >/dev/null; then aws() { command aws "$@" | tr -d '\r'; }; fi

[ "$(aws sts get-caller-identity --query Account --output text)" = "$ACCOUNT" ] || { echo "Wrong AWS account"; exit 1; }

say() { echo "== $*"; }

alb_arn() { aws elbv2 describe-load-balancers --names $NAME --query 'LoadBalancers[0].LoadBalancerArn' --output text 2>/dev/null || true; }
cert_arn() { aws acm list-certificates --query "CertificateSummaryList[?DomainName=='$DOMAIN'].CertificateArn | [0]" --output text; }

if [ "${1:-}" = "--https" ]; then
  CERT=$(cert_arn)
  STATUS=$(aws acm describe-certificate --certificate-arn "$CERT" --query Certificate.Status --output text)
  [ "$STATUS" = "ISSUED" ] || { echo "Certificate is $STATUS, not ISSUED yet: add the validation CNAME and wait."; exit 1; }
  ALB=$(alb_arn)
  TG=$(aws elbv2 describe-target-groups --names $NAME --query 'TargetGroups[0].TargetGroupArn' --output text)
  if ! aws elbv2 describe-listeners --load-balancer-arn "$ALB" --query 'Listeners[].Port' --output text | grep -qw 443; then
    aws elbv2 create-listener --load-balancer-arn "$ALB" --protocol HTTPS --port 443 \
      --ssl-policy ELBSecurityPolicy-TLS13-1-2-2021-06 --certificates CertificateArn="$CERT" \
      --default-actions Type=forward,TargetGroupArn="$TG" >/dev/null
  fi
  L80=$(aws elbv2 describe-listeners --load-balancer-arn "$ALB" --query 'Listeners[?Port==`80`].ListenerArn | [0]' --output text)
  aws elbv2 modify-listener --listener-arn "$L80" \
    --default-actions 'Type=redirect,RedirectConfig={Protocol=HTTPS,Port=443,StatusCode=HTTP_301}' >/dev/null
  say "HTTPS on: https://$DOMAIN (HTTP redirects to it)"
  exit 0
fi

TAG=${1:-$(aws ecr describe-images --repository-name $NAME --query 'sort_by(imageDetails,&imagePushedAt)[-1].imageTags[0]' --output text)}
IMAGE=$REPO:$TAG
SECRET_ARN=$(aws secretsmanager describe-secret --secret-id $SECRET_ID --query ARN --output text)
say "Image $IMAGE"

VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SUBNETS=$(aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC Name=default-for-az,Values=true \
  --query 'Subnets[?AvailabilityZone!=`us-east-1e`].SubnetId' --output text | tr '\t' ' ')
SUBNETS_CSV=$(echo $SUBNETS | tr ' ' ',')
say "VPC $VPC, subnets $SUBNETS"

sg() { # name description
  local id
  id=$(aws ec2 describe-security-groups --filters Name=vpc-id,Values=$VPC Name=group-name,Values="$1" --query 'SecurityGroups[0].GroupId' --output text)
  if [ "$id" = "None" ]; then
    id=$(aws ec2 create-security-group --vpc-id $VPC --group-name "$1" --description "$2" --query GroupId --output text)
  fi
  echo "$id"
}
ALB_SG=$(sg $NAME-alb "deeptrack-api load balancer: HTTP/HTTPS from anywhere")
TASK_SG=$(sg $NAME-task "deeptrack-api task: port 8080 from its load balancer only")
for port in 80 443; do
  aws ec2 authorize-security-group-ingress --group-id $ALB_SG --protocol tcp --port $port --cidr 0.0.0.0/0 >/dev/null 2>&1 || true
done
aws ec2 authorize-security-group-ingress --group-id $TASK_SG --protocol tcp --port 8080 --source-group $ALB_SG >/dev/null 2>&1 || true
say "Security groups $ALB_SG (ALB), $TASK_SG (task)"

ROLE=$NAME-ecs-execution
if ! aws iam get-role --role-name $ROLE >/dev/null 2>&1; then
  aws iam create-role --role-name $ROLE --description "deeptrack-api ECS tasks: pull the image, write logs, read the API key secret" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name $ROLE --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
  sleep 10
fi
aws iam put-role-policy --role-name $ROLE --policy-name read-api-key-secret --policy-document \
  "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"secretsmanager:GetSecretValue\",\"Resource\":\"$SECRET_ARN\"}]}"
ROLE_ARN=arn:aws:iam::$ACCOUNT:role/$ROLE
say "Execution role $ROLE_ARN"

aws logs create-log-group --log-group-name /ecs/$NAME >/dev/null 2>&1 || true
aws logs put-retention-policy --log-group-name /ecs/$NAME --retention-in-days 30
aws ecs describe-clusters --clusters $NAME --query 'clusters[?status==`ACTIVE`].clusterName' --output text | grep -q $NAME \
  || aws ecs create-cluster --cluster-name $NAME >/dev/null
say "Cluster $NAME"

TASKDEF_FILE=$(mktemp)
# Git Bash on Windows: the AWS CLI needs a Windows path.
TASKDEF_URI=file://$(command -v cygpath >/dev/null && cygpath -m "$TASKDEF_FILE" || echo "$TASKDEF_FILE")
${PYTHON:-python3} - "$IMAGE" "$SECRET_ARN" "$ROLE_ARN" > "$TASKDEF_FILE" <<'EOF'
import json, sys
image, secret, role = sys.argv[1:4]
print(json.dumps({
  "family": "deeptrack-api",
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "2048",
  "memory": "4096",
  "runtimePlatform": {"cpuArchitecture": "X86_64", "operatingSystemFamily": "LINUX"},
  "executionRoleArn": role,
  "containerDefinitions": [{
    "name": "api",
    "image": image,
    "essential": True,
    "portMappings": [{"containerPort": 8080, "protocol": "tcp"}],
    "environment": [{"name": "PORT", "value": "8080"}],
    "secrets": [{"name": "DEEPTRACK_API_KEYS", "valueFrom": secret}],
    "logConfiguration": {"logDriver": "awslogs", "options": {
      "awslogs-group": "/ecs/deeptrack-api", "awslogs-region": "us-east-1", "awslogs-stream-prefix": "api"}},
  }],
}))
EOF
TASKDEF=$(aws ecs register-task-definition --cli-input-json "$TASKDEF_URI" --query taskDefinition.taskDefinitionArn --output text)
rm -f "$TASKDEF_FILE"
say "Task definition $TASKDEF"

ALB=$(alb_arn)
if [ -z "$ALB" ] || [ "$ALB" = "None" ]; then
  ALB=$(aws elbv2 create-load-balancer --name $NAME --type application --scheme internet-facing \
    --subnets $SUBNETS --security-groups $ALB_SG --query 'LoadBalancers[0].LoadBalancerArn' --output text)
  # Video uploads and model warm-up can be slow.
  aws elbv2 modify-load-balancer-attributes --load-balancer-arn "$ALB" --attributes Key=idle_timeout.timeout_seconds,Value=120 >/dev/null
fi
ALB_DNS=$(aws elbv2 describe-load-balancers --load-balancer-arns "$ALB" --query 'LoadBalancers[0].DNSName' --output text)
TG=$(aws elbv2 describe-target-groups --names $NAME --query 'TargetGroups[0].TargetGroupArn' --output text 2>/dev/null || true)
if [ -z "$TG" ] || [ "$TG" = "None" ]; then
  TG=$(aws elbv2 create-target-group --name $NAME --protocol HTTP --port 8080 --vpc-id $VPC --target-type ip \
    --health-check-path /v1/health --health-check-interval-seconds 15 --healthy-threshold-count 2 \
    --unhealthy-threshold-count 5 --matcher HttpCode=200 --query 'TargetGroups[0].TargetGroupArn' --output text)
fi
aws elbv2 describe-listeners --load-balancer-arn "$ALB" --query 'Listeners[].Port' --output text | grep -qw 80 \
  || aws elbv2 create-listener --load-balancer-arn "$ALB" --protocol HTTP --port 80 --default-actions Type=forward,TargetGroupArn="$TG" >/dev/null
say "Load balancer $ALB_DNS"

NET="awsvpcConfiguration={subnets=[$SUBNETS_CSV],securityGroups=[$TASK_SG],assignPublicIp=ENABLED}"
if aws ecs describe-services --cluster $NAME --services $NAME --query 'services[?status==`ACTIVE`].serviceName' --output text | grep -q $NAME; then
  aws ecs update-service --cluster $NAME --service $NAME --task-definition "$TASKDEF" --force-new-deployment >/dev/null
  say "Service updated"
else
  aws ecs create-service --cluster $NAME --service-name $NAME --task-definition "$TASKDEF" --desired-count 1 \
    --launch-type FARGATE --network-configuration "$NET" --health-check-grace-period-seconds 180 \
    --load-balancers targetGroupArn="$TG",containerName=api,containerPort=8080 \
    --deployment-configuration 'deploymentCircuitBreaker={enable=true,rollback=true},maximumPercent=200,minimumHealthyPercent=100' >/dev/null
  say "Service created"
fi

CERT=$(cert_arn)
if [ -z "$CERT" ] || [ "$CERT" = "None" ]; then
  CERT=$(aws acm request-certificate --domain-name $DOMAIN --validation-method DNS --query CertificateArn --output text)
  sleep 10
fi
say "Certificate $CERT: $(aws acm describe-certificate --certificate-arn "$CERT" --query Certificate.Status --output text)"
echo
echo "DNS records to add at Namecheap (deeptrack.io):"
aws acm describe-certificate --certificate-arn "$CERT" \
  --query 'Certificate.DomainValidationOptions[0].ResourceRecord.[Type,Name,Value]' --output text \
  | awk '{ sub(/\.deeptrack\.io\.$/, "", $2); print "  " $1 "  host " $2 "  ->  " $3 }'
echo "  CNAME  host api  ->  $ALB_DNS"
echo
echo "Then: bash infra/ecs/deploy.sh --https"
