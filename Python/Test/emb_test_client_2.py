import asyncio
import grpc
import uuid
from grpc_file import (
    embed_pb2,
    embed_pb2_grpc,
    health_pb2,
    health_pb2_grpc,
    model_pb2,
    model_pb2_grpc,
)
from google.protobuf.json_format import MessageToDict
import pprint
import os

GRPC_ADDR = "localhost:50053"
VM_GRPC_ADDR = "192.168.20.114:50053"
TEST_ENDPOINT = "http://127.0.0.1:8000/v1/embeddings"


async def embedding_text(
    texts: list,
    model_name: str,
    api_key: str = "",
    task_type: str = "query",
    target: str = GRPC_ADDR,
):
    trace_id = str(uuid.uuid4())
    masked_key = f"{api_key[:8]}..." if api_key and len(api_key) > 10 else "None"
    print(f"\n📤 [요청] 모델: '{model_name}' | Task: {task_type} | Key: {masked_key}")
    print(f"🎫 [Trace ID]: {trace_id}")
    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = embed_pb2_grpc.EmbeddingServiceStub(channel)
            request = embed_pb2.EmbedRequest(
                text=texts, model_name=model_name, api_key=api_key, task_type=task_type
            )
            metadata = (("x-trace-id", trace_id),)
            response = await stub.Embed(request, metadata=metadata)

            result_dict = MessageToDict(
                response,
                preserving_proto_field_name=True,
                use_integers_for_enums=False,
            )
            # pprint.pprint(f"\n📦 [DEBUG] Raw Proto Response:\n{result_dict}")

            vec_count = len(response.vector)
            if vec_count > 0:
                results = [f"✅ 성공! (총 {vec_count}건)"]
                for i, vec in enumerate(response.vector):
                    input_text = texts[i] if i < len(texts) else "Unknown"
                    vec_len = len(vec.values)
                    preview = vec.values[:5]
                    lastview = vec.values[-5:]
                    results.append(
                        f"  [{i+1}] '{input_text}' -> 차원: {vec_len}, 값(앞5개): {preview}, 값(뒤5개): {lastview}"
                    )
                return "\n".join(results)
            return "⚠️ 반환된 벡터가 없습니다."

    except grpc.RpcError as e:
        return f"❌ gRPC 에러: {e.details()}"
    except Exception as e:
        return f"❌ 클라이언트 에러: {str(e)}"


async def list_models(target: str = GRPC_ADDR):
    trace_id = str(uuid.uuid4())
    print(f"\n🔍 [요청] 전체 모델 현황 조회 중... (Trace ID: {trace_id})")
    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = model_pb2_grpc.ModelServiceStub(channel)
            metadata = (("x-trace-id", trace_id),)
            response = await stub.Search(model_pb2.SearchRequest(), metadata=metadata)
            print(f"\n📦 [DEBUG] Raw Proto Response:\n{response}")

            print(f"\n{'='*110}")
            print(
                f" {'MODEL NAME':<25} | {'TYPE':<10} | {'STATUS':<15}     | {'DIM':<5} | {'DEV':<5} | {'REGISTERED AT'}"
            )
            print(f"{'-'*110}")

            if not response.models:
                print("   (등록된 모델 없음)")

            for m in response.models:
                if m.status == "활성화":
                    status_str = "🟢 활성화    "
                elif m.status == "오류":
                    status_str = "🔴 오류"
                elif m.status.startswith("Registered"):
                    status_str = "🟡 Registered"
                elif m.status.startswith("Ready"):
                    status_str = "🔵 Ready(Stateless)"
                else:
                    status_str = f"⚪ {m.status}"

                reg_time = (
                    m.registered_at.split(".")[0] if m.registered_at != "-" else "-"
                )

                model_type = getattr(m, "type", "Local")

                if model_type == "External":
                    device_str = "-"
                else:
                    device_str = "GPU" if getattr(m, "device", 0) == 1 else "CPU"

                print(
                    f" {m.model_name:<25} | {model_type:<10} | {status_str:<15} | {m.vector_dimension:<5} | {device_str:<5} | {reg_time}"
                )
            print(f"{'='*110}\n")

    except Exception as e:
        print(f"❌ 에러 발생: {e}")


async def load_model(model_name: str, target: str = GRPC_ADDR):
    trace_id = str(uuid.uuid4())
    print(f"\n📥 [요청] 로컬 모델 로드 시도: {model_name} (Trace ID: {trace_id})")

    use_gpu = input("👉 GPU를 사용하시겠습니까? (y/n, 기본값 n): ").strip().lower()
    device = 1 if use_gpu == "y" else 0

    device_name = "GPU" if device == 1 else "CPU"
    print(f"   Target Device: {device_name}")

    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = model_pb2_grpc.ModelServiceStub(channel)

            request = model_pb2.LoadRequest(model_name=model_name, device=device)
            metadata = (("x-trace-id", trace_id),)
            response = await stub.Load(request, metadata=metadata)
            print(f"\n📦 [DEBUG] Raw Proto Response:\n{response}")

            if response.code == 200:
                print(f"✅ 로드 성공: {response.message}")
            else:
                print(f"❌ 로드 실패 ({response.code}): {response.message}")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")


async def register_model(target: str = GRPC_ADDR):
    print("\n📝 [등록] 외부 모델(API) 등록 정보 입력")

    name = input("👉 모델 이름 (예: openai-emb): ").strip()
    if not name:
        return

    endpoint = input(
        "👉 API Endpoint (예: https://api.openai.com/v1/embeddings): "
    ).strip()

    dim_str = input("👉 벡터 차원 (예: 1536): ").strip()
    try:
        dim = int(dim_str)
    except ValueError:
        print("❌ 차원은 숫자여야 합니다.")
        return
    trace_id = str(uuid.uuid4())
    print(f"\n🚀 등록 요청 중... ({name}) (Trace ID: {trace_id})")

    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = model_pb2_grpc.ModelServiceStub(channel)

            request = model_pb2.RegisterRequest(
                model_name=name,
                api_endpoint=endpoint,
                dimension=dim,
            )
            metadata = (("x-trace-id", trace_id),)
            response = await stub.Register(request, metadata=metadata)

            if response.code == 200:
                print(f"✅ 등록 성공: {response.message}")
            else:
                print(f"❌ 등록 실패 ({response.code}): {response.message}")

    except grpc.RpcError as e:
        print(f"❌ gRPC 에러: {e.details()} (혹시 proto 파일 재생성 하셨나요?)")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")


async def delete_model(model_name: str, target: str = GRPC_ADDR):
    trace_id = str(uuid.uuid4())
    print(f"\n🗑️ [요청] 모델 삭제 시도: {model_name} (Trace ID: {trace_id})")
    print("⚠️ 주의: 로컬 모델은 파일 삭제 / 외부 모델은 설정 제거됨")

    warning = input("진짜로 삭제하시겠습니까? (y/n): ")
    if warning.lower() != "y":
        print("취소되었습니다.")
        return

    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = model_pb2_grpc.ModelServiceStub(channel)
            request = model_pb2.DeleteRequest(model_name=model_name)
            metadata = (("x-trace-id", trace_id),)
            response = await stub.Delete(request, metadata=metadata)
            print(f"\n📦 [DEBUG] Raw Proto Response:\n{response}")

            if response.code == 200:
                print(f"✅ 삭제 성공: {response.message}")
            else:
                print(f"❌ 삭제 실패 ({response.code}): {response.message}")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")


async def health_check(target: str = GRPC_ADDR):
    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = health_pb2_grpc.HealthStub(channel)
            req = health_pb2.HealthCheckRequest(service="")
            res = await stub.Check(req)
            print(f"\n📦 [DEBUG] Raw Proto Response:\n{res}")
            print(
                "\n🏥 Server Health:",
                health_pb2.HealthCheckResponse.ServingStatus.Name(res.status),
            )
    except Exception as e:
        print(f"❌ 헬스 체크 실패: {e}")


async def test_adhoc_embedding(target: str = GRPC_ADDR):
    print(f"\n{'='*20} 🧪 Ad-hoc(즉석) 실행 모드 {'='*20}")
    print("※ 서버에 등록되지 않은 API를 URL 입력만으로 즉시 테스트합니다.\n")

    print("[필수] API Endpoint (Full URL)")
    print("  - OpenAI 예: https://api.openai.com/v1/embeddings")
    print("  - Cohere 예: https://api.cohere.com/v1/embed")
    endpoint = input("👉 Endpoint: ").strip()
    if not endpoint:
        print("❌ Endpoint는 필수입니다.")
        return

    model_name = input(
        "👉 모델명 (API가 요구하는 이름, 예: text-embedding-3-small): "
    ).strip()
    if not model_name:
        print("❌ 모델명은 필수입니다.")
        return

    env_key = os.getenv("OPENAI_API_KEY") or os.getenv("COHERE_API_KEY") or ""
    api_key = input(f"👉 API Key (엔터 시 환경변수 사용): ").strip()
    if not api_key:
        api_key = env_key

    text_input = input("👉 테스트할 텍스트: ").strip()
    if not text_input:
        return
    texts = [text_input]

    task_type = (
        input("👉 Task Type (query/document, 기본값: query): ").strip() or "query"
    )

    trace_id = str(uuid.uuid4())
    print(f"\n🚀 [Ad-hoc 요청] {endpoint} 로 전송 중... (Trace: {trace_id})")

    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = embed_pb2_grpc.EmbeddingServiceStub(channel)

            request = embed_pb2.EmbedRequest(
                text=texts,
                model_name=model_name,
                api_key=api_key,
                task_type=task_type,
                api_endpoint=endpoint,
            )

            metadata = (("x-trace-id", trace_id),)
            response = await stub.Embed(request, metadata=metadata)

            vec_count = len(response.vector)
            print(f"\n✅ 응답 도착! (총 {vec_count}개 벡터)")
            for i, vec in enumerate(response.vector):
                preview = vec.values[:5]
                print(f"  [{i+1}] 차원: {len(vec.values)} | 값: {preview}...")

    except grpc.RpcError as e:
        print(f"❌ gRPC 에러: {e.details()}")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")


async def main_menu():
    while True:
        print("\n" + "=" * 45)
        print("      🤖 임베딩 서버 통합 관리 클라이언트")
        print("=" * 45)
        print("1. ⚡️ 텍스트 임베딩 요청 (Embed)")
        print("2. 📋 전체 모델 목록 조회 (List)")
        print("3. 📥 로컬 모델 로드 (Load - Local)")
        print("4. 📝 외부 모델 등록 (Register - External)")
        print("5. 🗑️ 모델 삭제/언로드 (Delete)")
        print("6. 🏥 서버 헬스 체크 (Health)")
        print("7. 🆕 Ad-hoc 테스트 (URL 직접 입력) ")
        print("0. 종료")
        print("-" * 45)

        choice = input("선택 > ")

        if choice == "1":
            print("\n[Step 1] 사용할 모델명 입력")
            model_name = (
                input("👉 모델명 (기본값: qwen3-embedding-0.6b): ").strip()
                or "qwen3-embedding-0.6b"
            )

            print(f"\n[Step 2] 임베딩할 텍스트 입력")
            text_input = input("👉 텍스트 (콤마 구분): ").strip()

            if text_input:
                texts = [t.strip() for t in text_input.split(",") if t.strip()]
                print("\n[Step 3] 옵션 설정")
                api_key_input = input(f"👉 API Key : ").strip()

                task_type_input = input(
                    "👉 Task Type (query/document, 기본값: query): "
                ).strip()
                final_task_type = task_type_input if task_type_input else "query"

                print(
                    await embedding_text(
                        texts,
                        model_name,
                        api_key=api_key_input,
                        task_type=final_task_type,
                    )
                )
            else:
                print("⚠️ 텍스트가 없습니다.")

        elif choice == "2":
            await list_models()

        elif choice == "3":
            name = input("👉 로드할 로컬 모델명: ").strip()
            if name:
                await load_model(name)

        elif choice == "4":
            await register_model()

        elif choice == "5":
            name = input("👉 삭제할 모델명: ").strip()
            if name:
                await delete_model(name)

        elif choice == "6":
            await health_check()

        elif choice == "7":
            await test_adhoc_embedding()

        elif choice == "0":
            print("👋 프로그램을 종료합니다.")
            break
        else:
            print("⚠️ 잘못된 입력입니다.")


if __name__ == "__main__":
    try:
        asyncio.run(main_menu())
    except KeyboardInterrupt:
        print("\n강제 종료됨.")
