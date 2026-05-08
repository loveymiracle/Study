import weaviate

client = weaviate.connect_to_custom(
    http_host="192.168.0.72",
    http_port=8081,
    http_secure=False,
    grpc_host="192.168.0.72",
    grpc_port=50052,
    grpc_secure=False,
)

try:
    collection = client.collections.get("ModelInfo")
    response = collection.query.fetch_objects(limit=10)

    print(f"🧐 총 {len(response.objects)}건의 모델 명세 정보가 조회되었습니다.\n")

    for i, obj in enumerate(response.objects, 1):
        p = obj.properties

        m_type_val = p.get("model_type", 0)
        m_type_text = "외부 API (0)" if m_type_val == 0 else "내부 모델 (1)"

        print(f"{'='*80}")
        print(f"[{i}] 임베딩 모델: {p.get('model_name', 'N/A')}")
        print(f"{'-'*80}")
        print(f"🔹 기본 정보")
        print(f"   - 모델 타입    : {m_type_text}")
        print(f"   - 벡터 차원    : {p.get('model_dimension', 'N/A')}")
        print(f"   - 버전         : {p.get('model_version', p.get('version', 'N/A'))}")
        print(f"   - 디바이스     : {p.get('model_device', 'N/A')}")

        print(f"\n🔹 연동 및 분석 정보")
        print(f"   - 토크나이저   : {p.get('tokenizer', 'N/A')}")
        print(f"   - 엔드포인트   : {p.get('api', 'N/A')}")

        api_key = p.get("api_key", "N/A")
        if isinstance(api_key, str) and api_key != "N/A":
            masked_key = f"{api_key[:8]}****"
        else:
            masked_key = "N/A"
        print(f"   - 외부 키      : {masked_key}")

        print(f"\n🔹 메타 데이터")
        print(f"   - 상태         : {p.get('status', 'N/A')}")
        print(f"   - 생성 일시    : {p.get('creation_time', 'N/A')}")
        print(f"   - 설명         : {p.get('model_desc', 'N/A')}")
        print(f"{'='*80}\n")

except Exception as e:
    print(f"🚨 조회 실패: {e}")

finally:
    client.close()
