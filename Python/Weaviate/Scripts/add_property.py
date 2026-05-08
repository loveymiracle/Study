import weaviate
from weaviate.classes.config import Property, DataType

client = weaviate.connect_to_custom(
    http_host="192.168.0.72",
    http_port=8081,
    http_secure=False,
    grpc_host="192.168.0.72",
    grpc_port=50052,
    grpc_secure=False,
)

try:
    print("🔌 Weaviate 연결 성공. 스키마 업데이트를 시작합니다...")

    collection = client.collections.get("ModelInfo")

    collection.config.add_property(Property(name="tokenizer", data_type=DataType.TEXT))

    print("✅ ModelInfo 컬렉션에 'tokenizer' 속성이 성공적으로 추가되었습니다!")

except Exception as e:
    print(f"🚨 오류 발생: {e}")

finally:
    client.close()
    print("👋 클라이언트 연결이 안전하게 종료되었습니다.")
