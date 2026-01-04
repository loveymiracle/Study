import asyncio
import grpc
import json
import os
import pprint
import csv
from google.protobuf.json_format import ParseDict
from google.protobuf.json_format import MessageToDict
from google.protobuf import struct_pb2
from grpc_file import collection_pb2, collection_pb2_grpc, search_pb2, search_pb2_grpc
from datetime import datetime, timezone

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "collection_dummy.json")
now_utc = datetime.now(timezone.utc)


# async def run():

#     dummy_data = load_dummy_data()
#     if not dummy_data:
#         return
#     try:
#         target = "localhost:50052"
#         print(f"⏳서버 연결 : {target}...")
#     except Exception as e:
#         print(f"서버 연결 실패 : {e}")

#     async with grpc.aio.insecure_channel(target) as channel:
#         stub = collection_pb2_grpc.CollectionServiceStub(channel)

#         for item in dummy_data:
#             print(f"\n📤 데이터 전송 시도: {item['collection_name']}")

#             vc_json = item["vector_config"]

#             vector_config_msg = collection_pb2.VectorConfig(
#                 model_name=vc_json.get("model_name"),
#                 model_dimension=vc_json.get("model_dimension"),
#                 distance=vc_json.get("distance"),
#                 chunk_type=vc_json.get("chunk_type", 0),
#                 chunk_size=vc_json.get("chunk_size", 0),
#                 chunk_overlap=vc_json.get("chunk_overlap", 0),
#             )

#             properties_msg_list = []
#             for prop in item["properties"]:
#                 p_item = collection_pb2.Properties(
#                     name=prop["name"],
#                     data_type=prop["data_type"],
#                     description=prop["description"],
#                 )
#                 properties_msg_list.append(p_item)

#             request = collection_pb2.CreateRequest(
#                 collection_name=item["collection_name"],
#                 collection_desc=item["collection_desc"],
#                 collection_type=item.get("collection_type", 0),
#                 vector_config=vector_config_msg,
#                 properties=properties_msg_list,
#             )

#             try:
#                 response = await stub.Create(request)
#                 print(f"✅ 성공 여부: {response.success}")
#                 print(f"✅ 메시지: {response.message}")

#             except grpc.RpcError as e:
#                 print(f"❌ 전송 실패 (gRPC Error): {e.code()}")
#                 print(f"🧾 내용: {e.details()}")
VM_GRPC_ADDR = "192.168.20.114:50052"


def load_dummy_data():
    if not os.path.exists(DATA_PATH):
        print(f"❌ 데이터 파일이 없습니다: {DATA_PATH}")
        return []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


async def create_collection(data: dict, target: str = "localhost:50052"):
    # async def create_collection(data: dict, target: str = VM_GRPC_ADDR):
    c_name = data.get("collection_name", "Unknown")
    print(f"📤 [요청] 컬렉션 생성 시도: {c_name}")

    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = collection_pb2_grpc.CollectionServiceStub(channel)

            request = ParseDict(data, collection_pb2.CreateRequest())

            await stub.Create(request)

            return True, "생성 성공"

    except grpc.RpcError as e:
        return False, f"gRPC 통신 에러: {e.details()}"
    except Exception as e:
        return False, f"데이터 변환/기타 에러: {str(e)}"


async def create_run():
    dummy_data = load_dummy_data()

    if not dummy_data:
        print("❌ 테스트할 데이터가 없습니다.")
        return

    print(f"⏳ 테스트 시작 (데이터 {len(dummy_data)}개)")

    for item in dummy_data:
        success, message = await create_collection(item)

        if success:
            print(f"✅ Result: {success}")
            print(f"✅ Content: {message}")
        else:
            print(f"❌ 실패: {message}")
        print("-" * 40)


async def delete_collection(name: str, target: str = "localhost:50052"):
    c_name = name
    print(f"\n🗑️ [요청] 컬렉션 삭제 시도: {c_name}")

    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = collection_pb2_grpc.CollectionServiceStub(channel)

            request = collection_pb2.DeleteRequest(collection_name=name)

            await stub.Delete(request)

            return True, "삭제 성공"

    except grpc.RpcError as e:
        return None
    except Exception as e:
        return None


async def delete_run():
    print("-" * 50)
    print("--- 🗑️ 컬렉션 삭제 테스트 시작 ---")

    try:
        collection_to_delete = input(
            "삭제할 컬렉션 이름을 입력하세요 (예: TechDocuments): "
        )
    except EOFError:
        print("입력값이 없어 삭제를 취소합니다.")
        return

    if not collection_to_delete:
        print("컬렉션 이름이 입력되지 않아 삭제를 취소합니다.")
        return

    success, message = await delete_collection(collection_to_delete)

    if success:
        print(f"✅ 삭제 성공: {message}")
    else:
        print(f"❌ 삭제 실패: {message}")
    print("-" * 50)


async def search_collection(
    name: str, page: int, page_size: int, target: str = "localhost:50052"
) -> collection_pb2.SearchResponse:
    c_name = name
    print(f"\n🔍 [요청] 컬렉션 조회 시도: {c_name}, Page: {page}, Size: {page_size}")

    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = collection_pb2_grpc.CollectionServiceStub(channel)

            request = collection_pb2.SearchRequest(
                collection_name=name, page=page, page_size=page_size
            )

            response = await stub.Search(request)
            return response

    except grpc.RpcError as e:
        error_msg = f"gRPC 통신 에러: {e.details()}"
        print(f"[DEBUG] 🚨 클라이언트 RPC 에러 발생: {error_msg}")
        return None
    except Exception as e:
        error_msg = f"클라이언트 조회 에러: {str(e)}"
        print(f"[DEBUG] 🚨 클라이언트 일반 에러 발생: {error_msg}")
        return None


async def search_run():
    print("-" * 50)
    print("--- 🔍 컬렉션 조회 테스트 시작 ---")

    try:
        col_name = input("조회할 컬렉션 이름을 입력하세요 (예: ClassifyIntent): ")
        page_str = input("페이지 번호를 입력하세요 (예: 1): ")
        size_str = input("페이지 크기를 입력하세요 (예: 10): ")

        page = int(page_str) if page_str.isdigit() else 1
        page_size = int(size_str) if size_str.isdigit() else 10

    except EOFError:
        print("입력값이 없어 조회를 취소합니다.")
        return

    response = await search_collection(col_name, page, page_size)

    if response:
        print(f"✅ 조회 성공")
        # vec = response.data[0].vector
        # print(vec)
        pprint.pprint(response)
        print("\n--- 조회 결과 요약 ---")

        print(f"컬렉션 이름: {response.collection_name}")
        print(f"전체 항목 수: {response.total_count}")
        print(f"현재 페이지 / 크기: {response.page} / {response.page_size}")

        if response.data:
            print(f"\n--- 데이터 목록 ({len(response.data)}건) ---")
            for idx, item in enumerate(response.data, start=1):
                props = dict(item.properties)

                print(f"\n[{idx}] 검색 결과")
                print(f"  프로퍼티 개수: {len(props)}")
                print(f"  프로퍼티 키: {list(props.keys())}")

                for k, v in props.items():
                    print(f"  {k}: {v}")

                if item.vector:
                    print(f"  벡터 크기: {len(item.vector)} (float)")
        else:
            print("조회된 항목이 없습니다.")

        print("--------------------\n")
    else:
        print(f"❌ 조회 실패")
    print("-" * 50)


async def search_property(
    name: str, key: str, target: str = "localhost:50052"
) -> collection_pb2.SearchPropertyResponse:
    print(f"\n🔍 [요청] 컬렉션 속성 조회 시도: {name}, Key : {key}")

    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = collection_pb2_grpc.CollectionServiceStub(channel)

            request = collection_pb2.SearchPropertyRequest(
                collection_name=name, primary_key=key
            )

            response = await stub.SearchProperty(request)
            return response
    except grpc.RpcError as e:
        error_msg = f"gRPC 통신 에러: {e.details()}"
        print(f"[DEBUG] 🚨 클라이언트 RPC 에러 발생: {error_msg}")
        return None
    except Exception as e:
        error_msg = f"클라이언트 조회 에러: {str(e)}"
        print(f"[DEBUG] 🚨 클라이언트 일반 에러 발생 : {error_msg}")
        return None


async def search_property_run():
    print("-" * 50)
    print("--- 🔍 컬렉션 속성(id) 조회 테스트 시작 ---")

    try:
        col_name = input("조회할 컬렉션 이름 입력하세요: ")
        pk_id = input("속성 조회에 사용할 키 입력하세요: ")

    except EOFError:
        print("입력값이 없어 취소")
        return

    response = await search_property(col_name, pk_id)

    if response:
        print(f"<✅ 조회 성공 >")
        print("\n--- 조회 결과 요약 ---")

        print(f"컬렉션 이름: {response.collection_name}")
        print(f"조회 기준 id: {response.primary_key}")

        if response.data:

            for idx, item in enumerate(response.data, start=1):
                props = dict(item.properties)

                print(f"\n[{idx}] 검색 결과")
                print(f"  프로퍼티 개수: {len(props)}")
                print(f"  프로퍼티 값: {list(props.keys())}")

                # for k, v in props.items():
                #     print(f"  {k}: {v}")

                if hasattr(item, "vector") and isinstance(item.vector, list):
                    print(f"  벡터 크기: {len(item.vector)} (float)")
        else:
            print("조회된 항목이 없습니다.")

        print("--------------------\n")
    else:
        print(f"❌ 조회 실패")
    print("-" * 50)


async def search_list(
    target: str = "localhost:50052",
) -> collection_pb2.SearchListResponse:
    print(f"\n🔍 [요청] 컬렉션 리스트 조회 시도")

    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = collection_pb2_grpc.CollectionServiceStub(channel)

            request = collection_pb2.SearchListRequest()

            response = await stub.SearchList(request)
            return response
    except grpc.RpcError as e:
        error_msg = f"gRPC 통신 에러: {e.details()}"
        print(f"[DEBUG] 🚨 클라이언트 RPC 에러 발생: {error_msg}")
        return None
    except Exception as e:
        error_msg = f"클라이언트 조회 에러: {str(e)}"
        print(f"[DEBUG] 🚨 클라이언트 일반 에러 발생 : {error_msg}")
        return None


async def search_list_run():
    print("-" * 50)
    print("--- 🔍 컬렉션 리스트 조회 테스트 시작 ---")

    response = await search_list()

    if response:
        print(f"✅ 조회 성공")
        pprint.pprint(response)
        print("\n--- 조회 결과 요약 ---")

        proto_to_json = MessageToDict(
            response,
            preserving_proto_field_name=True,  # 카멜 변경
            always_print_fields_with_no_presence=True,  # 0 출력
        )
        pprint.pprint(proto_to_json)

        for msg in response.collection_list:
            print(f"컬렉션 이름 : {msg.collection_name}")
            print(f"컬렉션 타입 : {msg.collection_type}")

        print("--------------------\n")
    else:
        print(f"❌ 조회 실패")
    print("-" * 50)


async def search_detailed(
    name: str, target: str = "localhost:50052"
) -> collection_pb2.SearchDetailedResponse:
    print(f"\n🔍 [요청] 컬렉션 리스트 조회 시도")

    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = collection_pb2_grpc.CollectionServiceStub(channel)

            request = collection_pb2.SearchDetailedRequest(collection_name=name)

            response = await stub.SearchDetailed(request)
            return response
    except grpc.RpcError as e:
        error_msg = f"gRPC 통신 에러: {e.details()}"
        print(f"[DEBUG] 🚨 클라이언트 RPC 에러 발생: {error_msg}")
        return None
    except Exception as e:
        error_msg = f"클라이언트 조회 에러: {str(e)}"
        print(f"[DEBUG] 🚨 클라이언트 일반 에러 발생 : {error_msg}")
        return None


async def search_detailed_run():
    print("-" * 50)
    print("--- 🔍 컬렉션 상세 조회 테스트 시작 ---")

    try:
        col_name = input(
            "조회할 컬렉션(사용자가 만든, CollectionInfo(x)) 이름 입력하세요: "
        )

    except EOFError:
        print("입력값이 없어 취소")
        return

    response = await search_detailed(col_name)

    if response:
        print(f"✅ 조회 성공")
        print("\n--- 조회 결과 요약 ---")

        proto_to_json = MessageToDict(
            response,
            preserving_proto_field_name=True,
            always_print_fields_with_no_presence=True,
        )
        pprint.pprint(proto_to_json)

        print("--------------------\n")
    else:
        print(f"❌ 조회 실패")
    print("-" * 50)


async def search_schema(
    name: str, target: str = "localhost:50052"
) -> collection_pb2.SearchSchemaResponse:
    print(f"\n🔍 [요청] 컬렉션 스키마 조회 시도")

    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = collection_pb2_grpc.CollectionServiceStub(channel)

            request = collection_pb2.SearchSchemaRequest(collection_name=name)

            response = await stub.SearchSchema(request)
            return response
    except grpc.RpcError as e:
        error_msg = f"gRPC 통신 에러: {e.details()}"
        print(f"[DEBUG] 🚨 클라이언트 RPC 에러 발생: {error_msg}")
        return None
    except Exception as e:
        error_msg = f"클라이언트 조회 에러: {str(e)}"
        print(f"[DEBUG] 🚨 클라이언트 일반 에러 발생 : {error_msg}")
        return None


async def search_schema_run():
    print("-" * 50)
    print("--- 🔍 컬렉션 스키마 조회 테스트 시작 ---")

    try:
        col_name = input("조회할 컬렉션 이름 입력하세요: ")

    except EOFError:
        print("입력값이 없어 취소")
        return

    response = await search_schema(col_name)

    if response:
        print(f"✅ 조회 성공")
        print("\n--- 조회 결과 요약 ---")

        proto_to_json = MessageToDict(
            response,
            preserving_proto_field_name=True,
        )
        pprint.pprint(proto_to_json)

        print("--------------------\n")
    else:
        print(f"❌ 조회 실패")
    print("-" * 50)


async def update_desc(
    name: str, desc: str, target: str = "localhost:50052"
) -> collection_pb2.UpdateDescResponse:
    print(f"\n🔍 [요청] 컬렉션 설명 수정 시도")

    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = collection_pb2_grpc.CollectionServiceStub(channel)

            request = collection_pb2.UpdateDescRequest(
                collection_name=name, collection_desc=desc
            )

            await stub.UpdateDesc(request)
            return True
    except grpc.RpcError as e:
        print(f"[DEBUG] 🚨 gRPC 통신 에러: {e.details()} (Code: {e.code()})")
        return None
    except Exception as e:
        print(f"[DEBUG] 🚨 클라이언트 일반 에러: {str(e)}")
        return None


async def update_desc_run():
    print("-" * 50)
    print("--- 🔍 컬렉션 스키마 조회 테스트 시작 ---")

    try:
        col_name = input("수정할 컬렉션 이름 입력하세요: ")
        desc = input("수정할 내용 입력하세요.")

    except EOFError:
        print("입력값이 없어 취소")
        return

    response = await update_desc(col_name, desc)

    if response:
        print(f"✅ 수정 성공")

        print("--------------------\n")
    else:
        print(f"❌ 조회 실패")
    print("-" * 50)


async def batch(
    name: str, documents: list, target: str = "localhost:50052"
) -> collection_pb2.BatchResponse:
    print(f"\n📦 [요청] 컬렉션 적재 시도: {name} (총 {len(documents)}건)")

    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = collection_pb2_grpc.CollectionServiceStub(channel)

            proto_objects = []
            for doc in documents:
                s = struct_pb2.Struct()
                s.update(doc)
                proto_objects.append(s)

            request = collection_pb2.BatchRequest(
                collection_name=name, data_objects=proto_objects
            )

            response = await stub.Batch(request)
            return response

    except grpc.RpcError as e:
        print(f"[DEBUG] 🚨 gRPC 통신 에러: {e.details()} (Code: {e.code()})")
        return None
    except Exception as e:
        print(f"[DEBUG] 🚨 클라이언트 일반 에러: {str(e)}")
        return None


async def batch_run():
    print("-" * 50)
    print("--- 🔍 CSV 데이터 적재 테스트 시작 ---")

    csv_file_path = "data/1k_products.csv"

    if not os.path.exists(csv_file_path):
        print(f"❌ 파일이 없습니다: {csv_file_path}")
        return

    col_name = "Sample"

    documents = []
    print(f"📂 CSV 파일 읽는 중... ({csv_file_path})")

    try:
        with open(csv_file_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                documents.append(
                    {
                        "name": row.get("name", ""),
                        "url": row.get("url", ""),
                        "description": row.get("description", ""),
                    }
                )

    except Exception as e:
        print(f"❌ CSV 읽기 실패: {e}")
        return

    if not documents:
        print("❌ 적재할 데이터가 없습니다.")
        return

    response = await batch(col_name, documents)

    if response:
        print(f"✅ 적재 완료!")
        print(f"  - 성공: {response.success_count}건")
        print(f"  - 실패: {response.failed_count}건")

        if response.failed_count > 0:
            print("  - [에러 샘플]")
            for err in response.error_messages[:3]:
                print(f"    * {err}")
    else:
        print(f"❌ 적재 실패 (서버 응답 없음)")

    print("-" * 50)


async def searchall(
    query: str,
    query_type: int,
    alpha: float,
    size: int,
    collection_names: list,
    vector: list,
    target: str = "localhost:50052",
) -> search_pb2.SearchAllResponse:
    query_map = {0: "Hybrid", 1: "VECTOR", 2: "BM25", 3: "text"}
    category = query_map.get(query_type, 3)
    print(f"\n📦 [요청] 검색: '{query}' ({category}) | 대상: {collection_names}")

    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = search_pb2_grpc.SearchServiceStub(channel)

            request = search_pb2.SearchAllRequest(
                query=query,
                query_type=query_type,
                alpha=alpha,
                size=size,
                collection_name=collection_names,
                vector=vector,
            )

            response = await stub.SearchAll(request)
            pprint.pprint(response)
            response_dict = MessageToDict(response)

            print("✅ 검색 완료!")

            print(json.dumps(response_dict, indent=2, ensure_ascii=False))

            return response_dict

    except grpc.RpcError as e:
        print(f"[DEBUG] 🚨 gRPC 통신 에러: {e.details()} (Code: {e.code()})")
        return None
    except Exception as e:
        print(f"[DEBUG] 🚨 클라이언트 일반 에러: {str(e)}")
        return None


async def searchall_run():
    print("-" * 50)
    print("--- 🔍 다중 컬렉션 검색 테스트 ---")

    try:
        query = input("검색어 (기본: 활동인구): ") or "활동인구"

        qt_input = input("타입 (0:Hybrid, 1:Vector, 2:BM25, 3:검색) [기본: 3]: ")
        query_type = int(qt_input) if qt_input else 3

        alpha_input = input("Alpha (0.0~1.0) [기본: 0.5]: ")
        alpha = float(alpha_input) if alpha_input else 0.5

        size_input = input("Limit [기본: 3]: ")
        size = int(size_input) if size_input else 3

        name_input = input("컬렉션 이름 (쉼표 구분) [기본: ClassifyIntent]: ")
        if not name_input:
            names = ["ClassifyIntent"]
        else:
            names = [n.strip() for n in name_input.split(",")]

        vec_input = input("벡터값 (예: [0.1, 0.2] / 없으면 엔터): ")
        vec = []
        if vec_input.strip():
            try:
                vec = json.loads(vec_input)
            except:
                print("⚠️ 벡터 형식이 잘못되었습니다. 빈 리스트로 보냅니다.")
                vec = []

    except Exception as e:
        print(f"❌ 입력값 처리 중 에러: {e}")
        return

    response = await searchall(query, query_type, alpha, size, names, vec)

    if response:
        print(f"✅ 검색 완료!")
        pprint.pprint(response)
    else:
        print(f"❌ 검색 실패")

    print("-" * 50)


async def test_graphql(
    gql_query: str, target: str = "localhost:50052"
) -> search_pb2.SemanticResponse:
    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = search_pb2_grpc.SearchServiceStub(channel)

            request = search_pb2.SemanticRequest(
                gql_query=gql_query,
            )

            response = await stub.SemanticSearch(request)
            result_dict = MessageToDict(response.result)

            print("✅ 실행 완료!")
            print(json.dumps(result_dict, indent=2, ensure_ascii=False))

            return result_dict

    except grpc.RpcError as e:
        print(f"[DEBUG] 🚨 gRPC 통신 에러: {e.details()} (Code: {e.code()})")
        return None
    except Exception as e:
        print(f"[DEBUG] 🚨 클라이언트 일반 에러: {str(e)}")
        return None


async def test_graphql_run():
    print("-" * 50)
    print("--- 🔍 GraphQL 검색 테스트 ---")

    try:
        gql_query = input("GraphQL :")

    except Exception as e:
        print(f"❌ 입력값 처리 중 에러: {e}")
        return

    response = await test_graphql(gql_query)

    if response:
        print(f"✅ 검색 완료!")
        pprint.pprint(response)
    else:
        print(f"❌ 검색 실패")

    print("-" * 50)


async def update_prop(
    collection_name: str,
    identifier: str,
    properties: dict,
    target: str = "localhost:50052",
) -> collection_pb2.UpdatePropResponse:
    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = collection_pb2_grpc.CollectionServiceStub(channel)

            request = collection_pb2.UpdatePropRequest(
                collection_name=collection_name,
                identifier=identifier,
                properties=properties,
            )

            response = await stub.UpdateProp(request)
            return response

    except grpc.RpcError as e:
        print(f"[DEBUG] 🚨 gRPC 통신 에러: {e.details()} (Code: {e.code()})")
        return None
    except Exception as e:
        print(f"[DEBUG] 🚨 클라이언트 일반 에러: {str(e)}")
        return None


async def update_prop_run():
    print("-" * 50)
    print("--- 🛠️ Update Model Info 테스트 ---")

    try:
        collection_name = input("target collection name (필수): ").strip()
        if not collection_name:
            print("❌ 컬렉션 이름은 필수입니다.")
            return
        identifier = input("Target Identifier (예: 모델명, 유저ID): ").strip()
        if not identifier:
            print("❌ 식별자는 필수입니다.")
            return

        properties = {}
        if collection_name == "ModelInfo":
            desc = input("New Description: ").strip()
            if desc:
                properties["model_desc"] = desc

            ver = input("New Version: ").strip()
            if ver:
                properties["model_version"] = ver
        elif collection_name == "MemberInfo":
            pw = input("New Password: ").strip()
            if pw:
                properties["pw"] = pw
            role = input("New Role: ").strip()
            if role:
                properties["role"] = role
        else:
            key = input("Custom Key: ").strip()
            val = input("Custom Value: ").strip()
            if key and val:
                properties[key] = val

        pprint.pprint(properties)
        if not properties:
            print("⚠️ 변경할 내용이 없어 전송 X")
            return

    except Exception as e:
        print(f"❌ 입력값 처리 중 에러: {e}")
        return

    response = await update_prop(collection_name, identifier, properties)

    if response is not None:
        print(f"\n✅ 수정 요청 성공! (Server returned success)")

        print(f"\n[1] Raw gRPC Object:")
        print(response)

        result_dict = MessageToDict(
            response,
            preserving_proto_field_name=True,
            use_integers_for_enums=False,
        )
        print(f"[2] Python Dictionary:")
        pprint.pprint(result_dict)
    else:
        print(f"\n❌ 수정 요청 실패")

    print("-" * 50)


async def delete_prop(
    collection_name: str, identifier: str, target: str = "localhost:50052"
) -> collection_pb2.DeletePropResponse:
    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = collection_pb2_grpc.CollectionServiceStub(channel)

            request = collection_pb2.DeletePropRequest(
                collection_name=collection_name, identifier=identifier
            )

            response = await stub.DeleteProp(request)
            return response
    except grpc.RpcError as e:
        print(f"[DEBUG] 🚨 gRPC 통신 에러: {e.details()} (Code: {e.code()})")
        return None
    except Exception as e:
        print(f"[DEBUG] 🚨 클라이언트 일반 에러: {str(e)}")
        return None


async def delete_prop_run():
    print("-" * 50)
    print("--- 🛠️ Update Model Info 테스트 ---")

    try:
        collection_name = input("target collection name (필수): ").strip()
        if not collection_name:
            print("❌ 컬렉션 이름은 필수입니다.")
            return
        identifier = input("Target Identifier (예: 모델명, 유저ID): ").strip()
        if not identifier:
            print("❌ 식별자는 필수입니다.")
            return
    except Exception as e:
        print(f"❌ 입력값 처리 중 에러: {e}")
        return

    response = await delete_prop(collection_name, identifier)

    if response is not None:
        print(f"\n✅ 수정 요청 성공! (Server returned success)")

        print(f"\n[1] Raw gRPC Object:")
        print(response)

        result_dict = MessageToDict(
            response,
            preserving_proto_field_name=True,
            use_integers_for_enums=False,
        )
        print(f"[2] Python Dictionary:")
        pprint.pprint(result_dict)
    else:
        print(f"\n❌ 수정 요청 실패")

    print("-" * 50)


async def enroll_obj(
    collection_name: str, properties: dict, target: str = "localhost:50052"
) -> collection_pb2.EnrollObjResponse:
    try:
        async with grpc.aio.insecure_channel(target) as channel:
            stub = collection_pb2_grpc.CollectionServiceStub(channel)

            request = collection_pb2.EnrollObjRequest(
                collection_name=collection_name, properties=properties
            )

            response = await stub.EnrollObj(request)
            return response
    except grpc.RpcError as e:
        print(f"[DEBUG] 🚨 gRPC 통신 에러: {e.details()} (Code: {e.code()})")
        return None
    except Exception as e:
        print(f"[DEBUG] 🚨 클라이언트 일반 에러: {str(e)}")
        return None


async def enroll_obj_run():
    print("-" * 50)
    print("--- 🛠️ 등록 테스트 ---")
    rfc3339 = now_utc.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

    try:
        collection_name = input("target collection name (필수): ").strip()
        if not collection_name:
            print("❌ 컬렉션 이름은 필수입니다.")
            return

        properties = {}
        if collection_name == "ModelInfo":
            m_name = input("Model Name (PK/필수): ").strip()
            if not m_name:
                print("❌ Model Name은 필수입니다.")
                return
            properties["model_name"] = m_name

            properties["model_desc"] = input("Model Desc: ").strip()
            properties["model_type"] = input("Model Type: ").strip()
            properties["model_device"] = input("Model Device: ").strip()
            properties["model_dimension"] = input("Model dimension: ").strip()
            properties["model_version"] = input("Model Version: ").strip()
            properties["creation_time"] = rfc3339
            properties["api_key"] = input("API Key: ").strip()
            properties["api"] = input("API(http://....): ").strip()

        elif collection_name == "MemberInfo":
            u_id = input("User ID (PK/필수): ").strip()
            if not u_id:
                print("❌ User ID는 필수입니다.")
                return
            properties["user_id"] = u_id

            properties["pw"] = input("Password: ").strip()
            properties["role"] = input("Role: ").strip()

        else:
            print("※ 범용 테스트 모드 (직접 Key/Value 입력, 종료하려면 엔터)")
            while True:
                k = input("Key: ").strip()
                if not k:
                    break
                v = input("Value: ").strip()
                properties[k] = v

    except Exception as e:
        print(f"❌ 입력값 처리 중 에러: {e}")
        return

    response = await enroll_obj(collection_name, properties)

    if response is not None:
        print(f"\n✅ 등록 요청 성공! (Server returned success)")

        print(f"\n[1] Raw gRPC Object:")
        print(response)

        result_dict = MessageToDict(
            response,
            preserving_proto_field_name=True,
            use_integers_for_enums=False,
        )
        print(f"[2] Python Dictionary:")
        pprint.pprint(result_dict)
    else:
        print(f"\n❌ 수정 요청 실패")

    print("-" * 50)


if __name__ == "__main__":
    # asyncio.run(create_run())
    # asyncio.run(delete_run())
    asyncio.run(search_run())
    # asyncio.run(search_property_run())
    # asyncio.run(search_list_run())
    # asyncio.run(search_detailed_run())
    # asyncio.run(search_schema_run())
    # asyncio.run(update_desc_run())
    # asyncio.run(batch_run())
    # asyncio.run(searchall_run())
    # asyncio.run(test_graphql_run())
    # asyncio.run(update_prop_run())
    # asyncio.run(delete_prop_run())
    # asyncio.run(enroll_obj_run())
