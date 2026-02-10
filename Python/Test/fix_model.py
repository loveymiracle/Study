from huggingface_hub import hf_hub_download

# 모델 ID와 저장할 폴더 경로
model_id = "jhgan/ko-sbert-nli"
save_dir = "./onnx_models/ko-sbert-nli"

print(f"📥 '{model_id}'의 가중치 파일(pytorch_model.bin) 다운로드 중...")

# 핵심 파일 1개만 딱 다운로드
hf_hub_download(
    repo_id=model_id,
    filename="pytorch_model.bin",
    local_dir=save_dir,
    local_dir_use_symlinks=False,
)

print("✅ 다운로드 완료! 이제 서버를 재시작하세요.")
