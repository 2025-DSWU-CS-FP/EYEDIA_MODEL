import subprocess

def run_script(script_path):
    print(f"\n🚀 실행 중: {script_path}")
    try:
        result = subprocess.run(["python", script_path], check=True)
        print(f"✅ 완료: {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"❗ 오류 발생: {script_path}")
        print(e)





if __name__ == "_main_":
    print("시작")
    run_script("scripts/fetch_text_and_build_faiss.py")
    run_script("scripts/crop_and_build_index.py")
    run_script("scripts/click_and_find_faiss.py")
    
    print("\n 완료")