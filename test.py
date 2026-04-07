import time
import requests

BASE_URL = "https://facedetectionsystem-staging.onrender.com"
API_KEY  = "dt_55cd38199f4d4c70af8fdd0d4edf59863e0de58f0f9a4af797f22083c7baab7d"
HEADERS  = {"X-API-Key": API_KEY}

def analyze_image(file_path: str):
    """Analyzes a single image for deepfake manipulation."""
    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/v1/image/predict",
            headers=HEADERS,
            files={"file": (file_path, f, "image/jpeg")},
        )
    resp.raise_for_status()
    return resp.json()

def analyze_video(file_path: str, poll_interval: int = 3):
    """Uploads a video and wait for the analysis result."""
    # upload
    with open(file_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/v1/video/predict/video",
            headers=HEADERS,
            files={"file": (file_path, f, "video/mp4")},
        )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    print(f"Job queued: {job_id}")

    # poll until done
    while True:
        resp   = requests.get(f"{BASE_URL}/v1/video/jobs/{job_id}", headers=HEADERS)
        status = resp.json()["status"]
        print(f"Status: {status}")

        if status == "done":
            return resp.json()["result"]
        if status == "error":
            raise Exception(resp.json()["error"])

        time.sleep(poll_interval)


def check_usage():
    resp = requests.get(f"{BASE_URL}/v1/client/usage/me", headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    # image
    result = analyze_image(r"C:\Users\LENOVO\Downloads\maid.jpg")
    print("Image result:", result)

    # video
    result = analyze_video(r"C:\Users\LENOVO\Downloads\grok-video-7f975d40-2a7a-4493-a5f1-b5b1a4979db4.mp4")
    print("Video result:", result)

    # usage
    usage = check_usage()
    print(f"Used: {usage['used_this_month']} / {usage['monthly_limit']}")