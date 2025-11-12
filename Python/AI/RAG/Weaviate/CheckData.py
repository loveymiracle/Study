import app.config  # 경로 및 경고 설정을 위해 가장 먼저 import
import sys
from pathlib import Path
import pprint
import weaviate.classes as wvc
import numpy as np
from weaviate.classes.query import Filter
import collections
import re
from weaviate.classes.aggregate import GroupByAggregate

from app.config.weaviateClient import get_weaviate_client

try:
    from kiwipiepy import Kiwi
except ImportError:
    print("❗️ 14번 [TermDef] 하이브리드 검색 기능이 비활성화됩니다.")
    Kiwi = None

SEARCH_TAGS = {"NNG", "NNP", "NP", "VV", "VA", "SL", "SH"}


def _get_kiwi_tokens(kiwi_analyzer: Kiwi, text: str) -> list[str]:
    """Kiwipie를 사용해 텍스트에서 검색용 토큰(형태소)을 추출합니다."""
    if not text or not kiwi_analyzer:
        return []
    try:
        tokens = kiwi_analyzer.tokenize(text)
        return [t.form for t in tokens if t.tag in SEARCH_TAGS]
    except Exception as e:
        print(f"Kiwipie 토큰화 중 오류 발생 : {e}")
        return []


def check_all_data(client):
    print("\n조회할 컬렉션을 선택하세요:")
    print("1. ClassifyIntent")
    print("2. IntentList")
    print("3. TermDef")
    choice = input("선택 (1, 2, 3): ").strip()

    if choice == "1":
        collection_name = "ClassifyIntent"
    elif choice == "2":
        collection_name = "IntentList"
    elif choice == "3":
        collection_name = "TermDef"
    else:
        print("❗️ 잘못된 선택입니다. 작업을 취소합니다.")
        return

    print(f"\n✅ '{collection_name}' 컬렉션의 전체 데이터 현황을 조회합니다.")
    print("✅ Weaviate 클라이언트 사용 시작!")

    try:
        if not client.collections.exists(collection_name):
            print(f"❗️ Collection '{collection_name}'이(가) 존재하지 않습니다.")
            return

        collection = client.collections.get(collection_name)

        # 1. 전체 문서 수 조회 (가장 정확한 방식)
        total_count = collection.aggregate.over_all(total_count=True).total_count
        print(f"✅ 전체 문서 수: {total_count}")

        # 2. .iterator()를 사용해 모든 객체 순회 및 직접 집계
        print("\n--- 전체 Category 목록 및 문서 수 (전체 순회) ---")

        # 데이터를 저장할 리스트 초기화
        category_list = []
        combo_list = []

        # .iterator()는 모든 데이터를 메모리에 올리지 않고 하나씩 가져옵니다.
        try:
            from tqdm import tqdm

            iterator = tqdm(
                collection.iterator(include_vector=False),
                total=total_count,
                desc="데이터 집계 중",
            )
        except ImportError:
            print("데이터 집계 중... (tqdm 라이브-러리가 설치되면 진행률이 표시됩니다)")
            iterator = collection.iterator(include_vector=False)

        for obj in iterator:
            category = obj.properties.get("category")
            intent = obj.properties.get("intent")

            # Category 리스트에 추가
            category_list.append(category)

            # Category와 Intent가 모두 유효한 경우 조합 리스트에 추가
            if category and intent:
                combo_list.append((category, intent))

        # 3. Category 집계 결과 출력
        category_counts = collections.Counter(category_list)
        valid_categories = []

        if None in category_counts:
            print(f"  - (Category 없음): {category_counts[None]}개")
            del category_counts[None]

        for category_name, count in sorted(category_counts.items()):
            print(f"  - {category_name}: {count}개")
            valid_categories.append(category_name)

        print(f"\n✅ 조회된 고유 Category: {sorted(valid_categories)}")

        # 4. Category + Intent 조합 결과 출력
        print("\n--- Category + Intent 조합별 문서 수 (전체 순회) ---")
        combo_counts = collections.Counter(combo_list)

        grouped_results = collections.defaultdict(list)
        for (category, intent), count in combo_counts.items():
            grouped_results[category].append((intent, count))

        for category, intent_list in sorted(grouped_results.items()):
            print(f"\n  📁 Category: {category}")
            for intent, count in sorted(intent_list):
                print(f"    - {intent}: {count}개")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")


# def check_all_data(client):
#     """전체 문서 수, 고유 카테고리 목록 및 샘플 데이터를 확인합니다."""

#     collection_name = "ClassifyIntent"
#     print("✅ Weaviate 클라이언트 사용 시작!")

#     try:
#         if not client.collections.exists(collection_name):
#             print(f"❗️ Collection '{collection_name}'이(가) 존재하지 않습니다.")
#             return

#         collection = client.collections.get(collection_name)

#         total_count_response = collection.aggregate.over_all(total_count=True)
#         total_count = total_count_response.total_count
#         print(f"✅ 전체 문서 수: {total_count}")

#         print("\n--- 전체 Category 목록 및 문서 수 ---")

#         response = collection.query.fetch_objects(
#             limit=100000, return_properties=["category", "intent"]
#         )

#         if not response.objects:
#             print("❗️ 조회된 데이터가 없습니다.")
#         else:
#             category_list = [
#                 obj.properties["category"]
#                 for obj in response.objects
#                 if "category" in obj.properties and obj.properties["category"]
#             ]

#             if not category_list:
#                 print("❗️ 모든 문서에 Category 값이 없거나 비어있습니다.")
#             else:
#                 category_counts = collections.Counter(category_list)

#                 categories = []
#                 for category_name, count in sorted(category_counts.items()):
#                     print(f"  - {category_name}: {count}개")
#                     categories.append(category_name)

#                 print(f"\n✅ 조회된 고유 Category: {categories}")

#             print("\n--- Category + Intent 조합별 문서 수 ---")

#         # 3. (category, intent) 튜플을 키로 사용하여 조합 리스트를 만듭니다.
#         combo_list = [
#             (obj.properties.get("category"), obj.properties.get("intent"))
#             for obj in response.objects
#             if obj.properties.get("category") and obj.properties.get("intent")
#         ]

#         if not combo_list:
#             print("❗️ Category와 Intent 조합을 만들 수 있는 데이터가 없습니다.")
#         else:
#             # 4. 조합 리스트의 개수를 계산하고 보기 좋게 출력합니다.
#             combo_counts = collections.Counter(combo_list)

#             # 카테고리별로 묶어서 출력하기 위한 딕셔너리
#             grouped_results = collections.defaultdict(list)
#             for (category, intent), count in combo_counts.items():
#                 grouped_results[category].append((intent, count))

#             # 묶인 결과를 정렬하여 출력
#             for category, intent_list in sorted(grouped_results.items()):
#                 print(f"\n  📁 Category: {category}")
#                 for intent, count in sorted(intent_list):
#                     print(f"    - {intent}: {count}개")

#         # 4. 데이터 5개를 가져옵니다. (항상 동일한 순서)
#         # print("\n--- 고정 데이터 샘플 5개 상세 정보 ---")
#         # query_response = collection.query.fetch_objects(limit=5, include_vector=True)

#         # if not query_response.objects:
#         #     print("❗️ 확인할 데이터가 없습니다.")
#         #     return

#         # # 5. 가져온 5개 객체를 순회하며 모든 정보를 출력합니다.
#         # for i, obj in enumerate(query_response.objects):
#         #     print(f"\n========== 고정 샘플 {i+1} ==========")
#         #     print(f"UUID: {obj.uuid}")
#         #     print("Properties:")
#         #     pprint.pprint(obj.properties)
#         #     if obj.vector and 'default' in obj.vector:
#         #         print(f"Vector Dimension: {len(obj.vector['default'])}")
#         #     else:
#         #         print("Vector: (저장된 벡터 없음)")

#     except Exception as e:
#         print(f"❌ 오류 발생: {e}")


def check_random_data(client):
    """
    무작위 벡터 검색을 이용해 효율적으로 5개 데이터 샘플을 확인합니다.
    """
    collection_name = "ClassifyIntent"
    print("\n--- 무작위 데이터 샘플 5개 상세 정보 (효율적인 방식) ---")

    try:
        collection = client.collections.get(collection_name)

        # 1. 모델의 벡터 차원 설정 (사용 중인 모델에 맞게 설정)
        # ko-sbert-sts 모델의 경우 768차원입니다.
        vector_dimension = 768

        # 2. 무작위 벡터 생성
        random_vector = np.random.randn(vector_dimension).tolist()

        # 3. 'near_vector'를 사용하여 무작위 벡터 주변의 객체 5개를 검색
        # 이 방식은 DB 인덱스를 활용하여 매우 빠릅니다.
        response = collection.query.near_vector(
            near_vector=random_vector,
            limit=5,
            include_vector=True,  # 벡터 데이터 포함하여 가져오기
        )

        # 4. 결과 출력
        for i, obj in enumerate(response.objects):
            print(f"\n========== 무작위 샘플 {i+1} ==========")
            print(f"UUID: {obj.uuid}")
            print("Properties:")
            pprint.pprint(obj.properties)

            # include_vector=True 이므로 obj.vector 에서 바로 확인 가능
            if obj.vector:
                vector_data = obj.vector.get("default")
                if vector_data:
                    print(f"Vector Dimension: {len(vector_data)}")
                else:
                    print("Vector: ('default' 벡터를 찾을 수 없음)")
            else:
                print("Vector: (저장된 벡터 없음)")

    except Exception as e:
        print(f"❌ 무작위 샘플 조회 중 오류 발생: {e}")


def search_by_text(client, query_text: str):
    """주어진 텍스트로 유사도 검색을 수행하고 상위 3개 결과를 출력합니다."""
    collection_name = "ClassifyIntent"
    print(f"\n--- 텍스트 검색 실행: '{query_text}' ---")

    try:
        if not client.collections.exists(collection_name):
            print(f"❗️ Collection '{collection_name}'이(가) 존재하지 않습니다.")
            return

        collection = client.collections.get(collection_name)

        response = collection.query.near_text(
            query=query_text,
            limit=3,
            return_metadata=wvc.query.MetadataQuery(certainty=True, distance=True),
        )

        if not response.objects:
            print("❗️ 검색 결과가 없습니다.")
            return

        print("✅ 검색 결과:")
        for i, obj in enumerate(response.objects):
            print(f"\n========== 검색 결과 {i+1} ==========")
            print("Properties:")
            pprint.pprint(obj.properties)
            print("검색 메타데이터:")
            pprint.pprint(obj.metadata)

    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")


def search_by_category(client, category_value: str):
    """주어진 category 값으로 검색하고 상위 10개 결과를 출력합니다."""
    collection_name = "ClassifyIntent"
    print(f"\n--- Category 검색 실행: '{category_value}' ---")

    try:
        if not client.collections.exists(collection_name):
            print(f"❗️ Collection '{collection_name}'이(가) 존재하지 않습니다.")
            return

        collection = client.collections.get(collection_name)

        # 🚀 v4 스타일 필터 정의
        # Filter.by_property("속성이름").equal("값") 형태로 필터를 생성합니다.
        category_filter = Filter.by_property("category").equal(category_value)

        # 🚀 collection.query.fetch_objects 사용 및 파라미터 이름 변경
        response = collection.query.fetch_objects(
            limit=1000,
            filters=category_filter,  # 'where' 대신 'filters' 사용
            # 'properties' 대신 'return_properties' 사용 (생략 시 모든 속성 반환)
            # return_properties=["*"]
        )

        if not response.objects:
            print("❗️ 검색 결과가 없습니다.")
            return

        print("✅ 검색 결과:")
        for i, obj in enumerate(response.objects):
            print(f"\n========== 검색 결과 {i+1} ==========")
            print("Properties:")
            pprint.pprint(obj.properties)
            print("검색 메타데이터:")
            pprint.pprint(obj.metadata)

    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")


def search_by_category_and_intent(client, category_value: str, intent_value: str):
    """주어진 category와 intent 값으로 동시에 검색하고 상위 10개 결과를 출력합니다."""
    collection_name = "ClassifyIntent"
    print(f"\n--- 검색 실행: category='{category_value}', intent='{intent_value}' ---")

    try:
        if not client.collections.exists(collection_name):
            print(f"❗️ Collection '{collection_name}'이(가) 존재하지 않습니다.")
            return

        collection = client.collections.get(collection_name)

        # 🚀 1. 두 개의 필터 조건을 `Filter.all_of`로 묶어줍니다.
        #    이것은 SQL의 'AND'와 동일하게 작동합니다.
        combined_filter = Filter.all_of(
            [
                Filter.by_property("category").equal(category_value),
                Filter.by_property("intent").equal(intent_value),
            ]
        )

        # 🚀 2. 생성된 통합 필터를 `filters` 인자로 전달합니다.
        response = collection.query.fetch_objects(limit=1000, filters=combined_filter)

        if not response.objects:
            print("❗️ 검색 결과가 없습니다.")
            return

        print("✅ 검색 결과:")
        all_messages = []
        for i, obj in enumerate(response.objects):
            print(f"\n========== 검색 결과 {i+1} ==========")
            print("Properties:")
            pprint.pprint(obj.properties)
            print("검색 메타데이터:")
            pprint.pprint(obj.metadata)

            msg = obj.properties.get("messages")
            if msg:
                all_messages.append(msg)

        if all_messages:
            print("\n✅ 전체 messages 모음:")
            # print(", ".join(all_messages))
            print("적재된 문서 갯수", len(all_messages))
            pprint.pprint(all_messages)

    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")
    finally:
        client.close()


def search_with_hybrid(client, query_text: str, alpha: float, threshold: float):
    """주어진 텍스트, alpha, threshold 값으로 하이브리드 검색을 수행합니다."""
    collection_name = "ClassifyIntent"
    print(
        f"\n--- 하이브리드 검색 실행: '{query_text}' (alpha={alpha}, threshold={threshold}) ---"
    )

    try:
        if not client.collections.exists(collection_name):
            print(f"❗️ Collection '{collection_name}'이(가) 존재하지 않습니다.")
            return

        collection = client.collections.get(collection_name)

        # 하이브리드 검색 실행
        response = collection.query.hybrid(
            query=query_text,
            alpha=alpha,
            limit=5,  # 상위 5개 결과 확인
            return_metadata=wvc.query.MetadataQuery(score=True),
        )

        if not response.objects:
            print("❗️ 검색 결과가 없습니다.")
            return

        print("✅ 검색 결과:")
        for i, obj in enumerate(response.objects):
            print(f"\n========== 검색 결과 {i+1} ==========")

            score = obj.metadata.score if obj.metadata else 0.0

            # 임계값 통과 여부 확인
            pass_status = "PASS" if score >= threshold else "FAIL"

            print(
                f"Status: [{pass_status}] (Score: {score:.4f} vs Threshold: {threshold})"
            )
            print("Properties:")
            pprint.pprint(obj.properties)
            print("검색 메타데이터:")
            pprint.pprint(obj.metadata)

    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")


def search_with_hybrid_and_category(
    client, query_text: str, category: str, alpha: float, threshold: float
):
    """주어진 텍스트, alpha, threshold 값으로 하이브리드 검색을 수행합니다."""
    collection_name = "ClassifyIntent"
    print(
        f"\n--- 하이브리드 검색 실행: '{query_text}, {category}' (alpha={alpha}, threshold={threshold}) ---"
    )

    try:
        if not client.collections.exists(collection_name):
            print(f"❗️ Collection '{collection_name}'이(가) 존재하지 않습니다.")
            return

        collection = client.collections.get(collection_name)
        filters = wvc.query.Filter.by_property("category").equal(category)

        # 하이브리드 검색 실행
        response = collection.query.hybrid(
            query=query_text,
            alpha=alpha,
            limit=5,  # 상위 5개 결과 확인
            filters=filters,
            return_metadata=wvc.query.MetadataQuery(score=True),
        )

        if not response.objects:
            print("❗️ 검색 결과가 없습니다.")
            return

        print("✅ 검색 결과:")
        for i, obj in enumerate(response.objects):
            print(f"\n========== 검색 결과 {i+1} ==========")

            score = obj.metadata.score if obj.metadata else 0.0

            # 임계값 통과 여부 확인
            pass_status = "PASS" if score >= threshold else "FAIL"

            print(
                f"Status: [{pass_status}] (Score: {score:.4f} vs Threshold: {threshold})"
            )
            print("Properties:")
            pprint.pprint(obj.properties)
            print("검색 메타데이터:")
            pprint.pprint(obj.metadata)

    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")


def diagnose_category_issue(client):
    """'category' 속성 관련 문제를 진단하기 위해 실제 데이터와 스키마 설정을 확인합니다."""

    collection_name = "ClassifyIntent"
    print(f"\n--- 🕵️ 'category' 속성 문제 진단 시작 ---")

    try:
        if not client.collections.exists(collection_name):
            print(f"❗️ Collection '{collection_name}'이(가) 존재하지 않습니다.")
            return

        collection = client.collections.get(collection_name)

        # 1. 실제 데이터 샘플 5개를 가져와 'category' 필드 값을 확인합니다.
        print("\n[1단계] 데이터 샘플 확인")
        response = collection.query.fetch_objects(limit=5)

        if not response.objects:
            print(" -> 데이터가 하나도 없습니다.")
        else:
            for i, obj in enumerate(response.objects):
                print(f"  - 샘플 {i+1} Properties:")
                pprint.pprint(obj.properties)

        # 2. 컬렉션의 스키마(설정) 정보를 가져와 'category' 속성의 인덱싱 상태를 확인합니다.
        print("\n[2단계] 'category' 속성 스키마(설정) 확인")
        config = collection.config.get()

        category_prop = None
        for prop in config.properties:
            if prop.name == "category":
                category_prop = prop
                break

        if category_prop:
            print(f" -> 'category' 속성 설정을 찾았습니다:")
            print(f"   - 이름(Name): {category_prop.name}")
            print(f"   - 데이터 타입(Data Type): {category_prop.data_type}")
            print(f"   - 토큰화(Tokenization): {category_prop.tokenization}")
            # index_filterable, index_searchable 속성 확인
            print(
                f"   - 필터 인덱싱 활성화 (index_filterable): {category_prop.index_filterable}"
            )
            print(
                f"   - 검색 인덱싱 활성화 (index_searchable): {category_prop.index_searchable}"
            )
        else:
            print(" -> 'category' 속성 설정을 찾을 수 없습니다.")

    except Exception as e:
        print(f"❌ 진단 중 오류 발생: {e}")


def delete_by_uuid(client):
    """지정된 UUID를 사용하여 ClassifyIntent 컬렉션에서 객체를 삭제합니다."""
    collection_name = "ClassifyIntent"
    print(f"\n--- UUID로 '{collection_name}' 컬렉션의 데이터 삭제 ---")

    try:
        uuid_to_delete = input("삭제할 객체의 UUID를 입력하세요: ").strip()
        if not uuid_to_delete:
            print("❗️ UUID가 입력되지 않았습니다. 작업을 취소합니다.")
            return

        if not client.collections.exists(collection_name):
            print(f"❗️ Collection '{collection_name}'이(가) 존재하지 않습니다.")
            return

        collection = client.collections.get(collection_name)

        collection.data.delete_by_id(uuid=uuid_to_delete)

        print(
            f"✅ '{collection_name}' 컬렉션에서 객체(UUID: {uuid_to_delete})를 성공적으로 삭제했습니다."
        )

    except Exception as e:
        print(f"❌ 삭제 중 오류 발생: {e}")


from weaviate.classes.query import Filter
import pprint


def search_for_deletion(client, property_name: str, property_value):
    """삭제하기 전에 어떤 객체들이 대상인지 검색하여 보여줍니다."""
    collection_name = "ClassifyIntent"
    print(
        f"\n--- [검색] '{property_name}'이(가) '{property_value}'인 객체를 찾습니다 ---"
    )

    try:
        collection = client.collections.get(collection_name)

        search_filter = Filter.by_property(property_name).equal(property_value)

        response = collection.query.fetch_objects(filters=search_filter, limit=5)

        if not response.objects:
            print("✅ 해당 조건에 맞는 객체가 없습니다. 삭제할 데이터가 없습니다.")
            return []

        print(
            f"🚨 총 {len(response.objects)}개의 객체가 삭제될 예정입니다. 내용을 확인하세요:"
        )
        for i, obj in enumerate(response.objects):
            print(f"\n--- [대상 {i+1}] ---")
            print(f"UUID: {obj.uuid}")
            pprint.pprint(obj.properties)

        return response.objects

    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")
        return []


def delete_by_property(client, property_name: str, property_value):
    """특정 속성 값과 일치하는 모든 객체를 삭제합니다."""
    collection_name = "ClassifyIntent"
    print(
        f"\n--- [삭제] '{property_name}'이(가) '{property_value}'인 객체를 삭제합니다 ---"
    )

    try:
        confirm = input(
            "❗️ 위 검색 결과에 나온 모든 객체를 정말로 삭제하시겠습니까? (y/n): "
        ).lower()
        if confirm != "y":
            print("✅ 작업을 취소했습니다.")
            return

        collection = client.collections.get(collection_name)

        delete_filter = Filter.by_property(property_name).equal(property_value)

        result = collection.data.delete_many(where=delete_filter)

        print("✅ 삭제 작업 완료!")
        pprint.pprint(result)  # 삭제 결과 출력

    except Exception as e:
        print(f"❌ 삭제 중 오류 발생: {e}")


def compare_search_scores(
    client, embedding_model, query_text: str, alpha: float, category: str
):
    """
    지정된 'category' 내에서, 'ClassifyIntent'와 'IntentList' 두 컬렉션에
    동일한 하이브리드 검색을 실행하여 점수와 내용을 비교합니다.
    """
    class_a_name = "ClassifyIntent"
    class_b_name = "IntentList"
    limit = 3

    print(f"\n--- 🔄 두 컬렉션 검색 결과 비교 ---")
    print(f"쿼리: '{query_text}'")
    print(f"Alpha: {alpha}")
    print(f"Category 필터: '{category}'")

    print("\n쿼리 텍스트를 ONNX 모델로 벡터 변환 중...")
    try:
        query_vector = embedding_model.embed_query(query_text)
        print(" -> 벡터 변환 완료!")
    except Exception as e:
        print(f"❌ 쿼리 벡터 변환 중 오류 발생: {e}")
        return

    def _search_and_display(collection_name: str):
        print("\n" + "-" * 20)
        print(f"🔍 '{collection_name}' 검색 결과")
        print("-" * 20)

        try:
            if not client.collections.exists(collection_name):
                print(f"❗️ Collection '{collection_name}'이(가) 존재하지 않습니다.")
                return

            collection = client.collections.get(collection_name)
            category_filter = wvc.query.Filter.by_property("category").equal(category)

            response = collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=alpha,
                limit=limit,
                filters=category_filter,
                return_metadata=wvc.query.MetadataQuery(score=True),
            )

            if not response.objects:
                print("  -> 해당 카테고리 내에 검색 결과가 없습니다.")
                return

            for i, obj in enumerate(response.objects):
                score = obj.metadata.score if obj.metadata else 0.0
                print(f"  [결과 {i+1}] Score: {score:.4f}")
                print(f"    - Category: {obj.properties.get('category')}")
                print(f"    - Intent: {obj.properties.get('intent')}")
                print(f"    - Messages: {obj.properties.get('messages')}")

        except Exception as e:
            print(f"❌ '{collection_name}' 검색 중 오류 발생: {e}")

    _search_and_display(class_a_name)
    _search_and_display(class_b_name)


def check_collection_schemas(client):
    """'ClassifyIntent'와 'IntentList' 컬렉션의 실제 설정을 비교하여 출력합니다."""

    class_a_name = "ClassifyIntent"
    class_b_name = "IntentList"

    print(f"\n--- 🕵️ 컬렉션 설정 비교: '{class_a_name}' vs '{class_b_name}' ---")

    def _get_and_print_config(collection_name: str):
        print("\n" + "=" * 30)
        print(f"'{collection_name}' 컬렉션 설정 확인")
        print("=" * 30)

        try:
            if not client.collections.exists(collection_name):
                print(f"❗️ Collection '{collection_name}'이(가) 존재하지 않습니다.")
                return

            collection = client.collections.get(collection_name)
            config = collection.config.get()

            print(f"✅ Vectorizer: {config.vectorizer}")
            print("✅ Properties:")
            for prop in config.properties:
                print(
                    f"  - {prop.name} (DataType: {prop.data_type}, Index Filterable: {prop.index_filterable})"
                )

        except Exception as e:
            print(f"❌ '{collection_name}' 설정 조회 중 오류 발생: {e}")

    _get_and_print_config(class_a_name)
    _get_and_print_config(class_b_name)


# def simulate_production_logic(client, embedding_model):
#     """
#     운영 환경의 intentProcessor 로직을 그대로 시뮬레이션하여 최종 점수를 확인합니다.
#     """
#     collection_name = "ClassifyIntent"  # 운영 로직은 ClassifyIntent를 사용
#     print(f"\n--- 🕵️ 운영 로직 시뮬레이션 (대상 컬렉션: {collection_name}) ---")

#     try:
#         user_input = input("테스트할 사용자 메시지 입력: ").strip()
#         page = input("현재 페이지(카테고리 필터용) 입력 (e.g., eap, gis): ").strip()

#         if not user_input or not page:
#             print("❗️ 메시지와 페이지를 모두 입력해야 합니다.")
#             return

#         # 1. 쿼리 벡터 생성
#         print("\n[Step 1] 쿼리 텍스트를 벡터로 변환 중...")
#         query_vector = embedding_model.embed_query(user_input)
#         print(" -> 완료.")

#         # 2. 동적 파라미터 설정
#         word_count = len(user_input.split())
#         if word_count <= 3:
#             search_alpha = 0.2
#             score_threshold = 0.8
#         else:
#             search_alpha = 0.6
#             score_threshold = 0.6
#         print(f"\n[Step 2] 동적 파라미터 설정 완료")
#         print(
#             f" -> 단어 수: {word_count}, Alpha: {search_alpha}, Threshold: {score_threshold}"
#         )

#         # 3. 병렬 검색 실행
#         print("\n[Step 3] Weaviate 병렬 검색 실행 (`com` 및 `{page}` 카테고리)")
#         collection = client.collections.get(collection_name)

#         def _blocking_search(filters=None):
#             response = collection.query.hybrid(
#                 query=user_input,
#                 vector=query_vector,
#                 alpha=search_alpha,
#                 limit=1,
#                 filters=filters,
#                 return_metadata=wvc.query.MetadataQuery(score=True),
#             )
#             if not response.objects:
#                 return None, None, None, 0.0

#             top_hit = response.objects[0]
#             metadata = top_hit.metadata
#             score = metadata.score if metadata else 0.0

#             if score >= score_threshold:
#                 properties = top_hit.properties
#                 intent = properties.get("intent")
#                 category = properties.get("category")
#                 matched_text = properties.get("messages")
#                 return intent, category, matched_text, score

#             return None, None, None, 0.0

#         # 동기식으로 순차 실행 (결과는 동일)
#         com_filter = wvc.query.Filter.by_property("category").equal("com")
#         page_filter = wvc.query.Filter.by_property("category").equal(page)

#         results = [
#             _blocking_search(filters=com_filter),
#             _blocking_search(filters=page_filter),
#         ]
#         print(" -> 검색 완료. 후보군 필터링 및 후처리 시작...")

#         # 4. 점수 후처리 (정확한 일치 보너스)
#         EXACT_MATCH_BONUS = 0.4
#         processed_results = []
#         for intent, category, matched_text, score in results:
#             if not intent:
#                 continue

#             current_score = score
#             # 공백 정규화 및 소문자 변환 후 비교
#             if (
#                 re.sub(r"\s+", " ", user_input).strip().lower()
#                 == re.sub(r"\s+", " ", matched_text).strip().lower()
#             ):
#                 current_score += EXACT_MATCH_BONUS

#             processed_results.append((intent, category, matched_text, current_score))

#         print("\n[Step 4] 후보군 점수 후처리 결과 (보너스 적용)")
#         if not processed_results:
#             print(" -> Threshold를 통과한 후보가 없습니다.")
#         for i, res in enumerate(processed_results):
#             print(
#                 f"  - 후보 {i+1}: Intent={res[0]}, Category={res[1]}, Score={res[3]:.4f}, Matched='{res[2]}'"
#             )

#         # 5. 최종 선택 로직
#         print("\n[Step 5] 최종 인텐트 선택")
#         if not processed_results:
#             best_result = ("ETC_GENERAL", page, user_input, 0.0)
#             print(" -> 후보 없음. ETC_GENERAL로 결정.")
#         else:
#             best_result = max(
#                 processed_results,
#                 key=lambda r: (
#                     r[3],  # 1순위: 최종 점수
#                     -abs(
#                         len(r[2]) - len(user_input)
#                     ),  # 2순위: 텍스트 길이 차이 (적을수록 좋음)
#                     r[1] == page,  # 3순위: 현재 페이지 카테고리 일치 여부
#                 ),
#             )
#             print(" -> 우선순위에 따라 최적 후보 선택 완료.")

#         print("\n" + "=" * 20 + " 최종 결과 " + "=" * 20)
#         print(f"  - 최종 Intent: {best_result[0]}")
#         print(f"  - 최종 Category: {best_result[1]}")
#         print(f"  - 최종 Score: {best_result[3]:.4f}")
#         print(f"  - 근거 Text: '{best_result[2]}'")
#         print("=" * 50)

#     except Exception as e:
#         print(f"❌ 시뮬레이션 중 오류 발생: {e}")


def _run_simulation_on_collection(
    client, embedding_model, user_input, page, collection_name, query_vector
):
    """(헬퍼 함수) 지정된 단일 컬렉션에 대해 운영 로직을 실행하고 최종 상위 3개 후보를 반환합니다."""

    print("\n" + "=" * 25)
    print(f"▶ 시뮬레이션 시작: [{collection_name}]")
    print("=" * 25)

    try:
        # 동적 파라미터 설정
        word_count = len(user_input.split())
        if word_count <= 3:
            search_alpha = 0.2
            score_threshold = 0.8
        else:
            search_alpha = 0.6
            score_threshold = 0.6
        print(
            f"[Step 1] 동적 파라미터: Alpha={search_alpha}, Threshold={score_threshold}"
        )

        # Weaviate 검색 실행
        collection = client.collections.get(collection_name)

        def _blocking_search(filters=None):
            response = collection.query.hybrid(
                query=user_input,
                vector=query_vector,
                alpha=search_alpha,
                limit=3,  # ### 변경: 후보 3개를 가져오도록 수정
                filters=filters,
                return_metadata=wvc.query.MetadataQuery(score=True),
            )

            candidates = []
            if not response.objects:
                return candidates

            for hit in response.objects:
                metadata = hit.metadata
                score = metadata.score if metadata else 0.0

                if score >= score_threshold:
                    properties = hit.properties
                    candidates.append(
                        (
                            properties.get("intent"),
                            properties.get("category"),
                            properties.get("messages"),
                            score,
                        )
                    )
            return candidates

        com_filter = wvc.query.Filter.by_property("category").equal("com")
        page_filter = wvc.query.Filter.by_property("category").equal(page)

        com_candidates = _blocking_search(filters=com_filter)
        page_candidates = _blocking_search(filters=page_filter)

        # 후보군 통합 및 중복 제거 (matched_text 기준)
        all_candidates = {}
        for candidate in com_candidates + page_candidates:
            matched_text = candidate[2]
            if matched_text not in all_candidates:
                all_candidates[matched_text] = candidate

        unique_candidates = list(all_candidates.values())
        print(
            f"[Step 2] Weaviate 검색 완료. 총 {len(unique_candidates)}개의 고유 후보군 후처리 시작..."
        )

        # 점수 후처리 (정확한 일치 보너스)
        EXACT_MATCH_BONUS = 0.4
        processed_results = []
        for intent, category, matched_text, score in unique_candidates:
            current_score = score
            if (
                matched_text
                and re.sub(r"\s+", " ", user_input).strip().lower()
                == re.sub(r"\s+", " ", matched_text).strip().lower()
            ):
                current_score += EXACT_MATCH_BONUS
                print(
                    f"  -> Exact Match 보너스 적용! (Intent: {intent}, Score: {score:.4f} -> {current_score:.4f})"
                )

            processed_results.append((intent, category, matched_text, current_score))

        # ### 변경: 최종 선택 로직을 정렬 후 상위 3개 선택으로 변경
        print("[Step 3] 우선순위에 따라 전체 후보 정렬...")
        if not processed_results:
            print(" -> 후보 없음.")
            return [("ETC_GENERAL", page, user_input, 0.0)]

        sorted_results = sorted(
            processed_results,
            key=lambda r: (r[3], -abs(len(r[2]) - len(user_input)), r[1] == page),
            reverse=True,
        )

        top_3_candidates = sorted_results[:3]
        print(f" -> 상위 {len(top_3_candidates)}개 후보 선택 완료.")
        return top_3_candidates

    except Exception as e:
        print(f"❌ [{collection_name}] 시뮬레이션 중 오류 발생: {e}")
        return [("ERROR", "ERROR", str(e), 0.0)]


def compare_production_logic(client, embedding_model):
    """
    사용자 입력을 받아 두 컬렉션에 대한 운영 로직 시뮬레이션을 각각 실행하고,
    최종 상위 3개 후보를 비교하여 요약 출력합니다.
    """
    print(f"\n--- 🕵️ 운영 로직 비교 시뮬레이션 (상위 3개 후보) ---")

    try:
        user_input = input("테스트할 사용자 메시지 입력: ").strip()
        page = input("현재 페이지(카테고리 필터용) 입력 (e.g., eap, gis): ").strip()

        if not user_input or not page:
            print("❗️ 메시지와 페이지를 모두 입력해야 합니다.")
            return

        print("\n[공통 작업] 쿼리 텍스트를 벡터로 변환 중...")
        query_vector = embedding_model.embed_query(user_input)
        print(" -> 완료.")

        class_a_name = "ClassifyIntent"
        class_b_name = "IntentList"

        result_a = _run_simulation_on_collection(
            client, embedding_model, user_input, page, class_a_name, query_vector
        )
        result_b = _run_simulation_on_collection(
            client, embedding_model, user_input, page, class_b_name, query_vector
        )

        # ### 변경: 최종 비교 요약 출력을 상위 3개 후보 모두 보여주도록 변경
        print("\n" + "=" * 30 + " 최종 비교 요약 " + "=" * 30)

        print(f"\n--- [결과 1] 컬렉션: {class_a_name} ---")
        if not result_a:
            print("  -> 최종 후보 없음.")
        else:
            for i, res in enumerate(result_a):
                print(f"  [후보 {i+1}]")
                print(f"    - 최종 Intent: {res[0]}")
                print(f"    - 최종 Category: {res[1]}")
                print(f"    - 최종 Score: {res[3]:.4f}")
                print(f"    - 근거 Text: '{res[2]}'")

        print(f"\n--- [결과 2] 컬렉션: {class_b_name} ---")
        if not result_b:
            print("  -> 최종 후보 없음.")
        else:
            for i, res in enumerate(result_b):
                print(f"  [후보 {i+1}]")
                print(f"    - 최종 Intent: {res[0]}")
                print(f"    - 최종 Category: {res[1]}")
                print(f"    - 최종 Score: {res[3]:.4f}")
                print(f"    - 근거 Text: '{res[2]}'")

        print("=" * 75)

    except Exception as e:
        print(f"❌ 비교 시뮬레이션 중 오류 발생: {e}")


def search_termdef_hybrid(
    client,
    embedding_model,
    kiwi_analyzer: Kiwi,
    query_text: str,
    alpha: float,
    threshold: float,
):
    """
    [TermDef] 컬렉션 대상.
    입력 텍스트를 임베딩(Dense)하고 Kiwipie(Sparse)로 분석하여 하이브리드 검색
    """
    collection_name = "TermDef"
    print(
        f"\n--- [TermDef] 하이브리드 검색: '{query_text}' (alpha={alpha}, threshold={threshold}) ---"
    )

    try:
        if not client.collections.exists(collection_name):
            print(f"❗️ Collection '{collection_name}'이(가) 존재하지 않습니다.")
            return

        collection = client.collections.get(collection_name)

        # --- 1. 쿼리 준비 (Dense + Sparse) ---
        print("[Step 1] 쿼리 준비 중...")
        # Dense 벡터 생성
        query_vector = embedding_model.embed_query(query_text)
        print(" -> Dense 벡터 생성 완료.")

        # Sparse 토큰 생성
        query_tokens_list = _get_kiwi_tokens(kiwi_analyzer, query_text)
        query_tokens_str = " ".join(query_tokens_list)
        print(f" -> Sparse 토큰 생성 완료: [{query_tokens_str}]")

        # --- 2. 하이브리드 검색 실행 ---
        print("[Step 2] Weaviate 하이브리드 검색 실행...")
        response = collection.query.hybrid(
            query=query_tokens_str,  # Sparse(BM25) 검색어 (Kiwipie 토큰)
            vector=query_vector,  # Dense 검색어 (임베딩 벡터)
            alpha=alpha,
            limit=5,
            # Sparse 검색이 'kiwi_tokens' 필드를 대상으로 하도록 명시
            query_properties=["kiwi_tokens"],
            return_metadata=wvc.query.MetadataQuery(score=True, explain_score=True),
            include_vector=True,
        )

        if not response.objects:
            print("❗️ 검색 결과가 없습니다.")
            return

        # --- 3. 결과 출력 ---
        print("\n✅ 검색 결과:")
        summary_list = []
        for i, obj in enumerate(response.objects):
            print(f"\n========== 검색 결과 {i+1} ==========")

            score = obj.metadata.score if obj.metadata else 0.0
            pass_status = "PASS" if score >= threshold else "FAIL"

            print(
                f"Status: [{pass_status}] (Score: {score:.4f} vs Threshold: {threshold})"
            )
            print("Properties:")
            # 'spec'은 JSON 문자열로 저장되어 있으므로 그대로 출력됩니다.
            pprint.pprint(obj.properties)
            print("검색 메타데이터:")
            pprint.pprint(obj.metadata)

            print("--- Vector (전체) ---")
            if obj.vector and "default" in obj.vector:
                vector_list = obj.vector["default"]
                print(f"  (차원: {len(vector_list)})")
                # pprint.pprint(obj.vector)
            else:
                print("  (벡터 값이 없습니다)")

            embedding_text = obj.properties.get("embedding_text", "N/A")
            summary_list.append((score, embedding_text, pass_status))

        print("\n" + "=" * 20 + " 💡 최종 요약 💡 " + "=" * 20)
        print(f"쿼리: '{query_text}' (Alpha: {alpha}, Threshold: {threshold})")

        if not summary_list:
            print(" -> 요약할 결과가 없습니다.")
            return

        for idx, (score, text, status) in enumerate(summary_list):
            text_snippet = (text[:70] + "...") if len(text) > 70 else text
            print(f"\n [요약 {idx+1}]")
            print(f"   Status: [{status}] (Score: {score:.4f})")
            print(f'   Text: "{text_snippet}"')

        print("=" * (46 + len(" 💡 최종 요약 💡 ")))

    except Exception as e:
        print(f"❌ 검색 중 오류 발생: {e}")


if __name__ == "__main__":
    client = get_weaviate_client()
    print("✅ Weaviate 클라이언트 연결 성공!")

    from app.config.openaiClient import get_embedding_model

    embedding_model = get_embedding_model()
    print("✅ 임베딩 모델 로드 성공!")

    kiwi_analyzer = None
    if Kiwi:
        try:
            kiwi_analyzer = Kiwi()
            print("✅ Kiwipie 형태소 분석기 로드 성공!")
        except Exception as e:
            print(f"❗️ Kiwipie 로드 실패: {e}")

    while True:
        try:
            print("\n" + "=" * 50)
            print("Weaviate 데이터 조회 스크립트")
            print("=" * 50)
            print("수행할 작업을 선택하세요:")
            print("1. 전체 데이터 현황 조회")
            print("2. 임의 데이터 조회")
            print("3. 검색어로 조회 (near_text)")
            print("4. 카테고리 조회")
            print("5. 카테고리 + 의도로 조회")
            print("6. 하이브리드 검색 (alpha, threshold 조절)")
            print("7. 하이브리드 검색 (category, alpha, threshold 조절)")
            print("8. 'category' 문제 진단")
            print("9. uuid로 삭제")
            print("10, 조회 후 삭제 (데이터 삭제 주의)")
            print("11. 두 컬렉션 검색 점수 비교")
            print("12. 컬렉션 설정(스키마) 비교")
            print("13. 운영 로직 시뮬레이션")
            print("14. [TermDef] 하이브리드 검색 (Kiwipie + Embedding)")
            print("Q. 종료")
            print("-" * 50)
            choice = input("선택 (1, 2, ..., 14 또는 Q): ")
            print("=" * 50)

            if choice == "1":
                print("\n--- 전체 데이터 조회합니다 ---")
                check_all_data(client)
            elif choice == "2":
                print("\n--- 임의 데이터 조회합니다 ---")
                check_random_data(client)
            elif choice == "3":
                print("\n--- 검색어 조회합니다 ---")
                search_query = input("검색어 입력: ")
                search_by_text(client, search_query)
            elif choice == "4":
                print("\n--- 카테고리 별로 조회합니다 ---")
                category = input("카테고리 입력: ")
                search_by_category(client, category)
            elif choice == "5":
                print("\n--- 카테고리 + 의도로 조회합니다 ---(ex: com REQ_TERMDEF)")
                category, intent = input(
                    "category와 intent 입력 (공백으로 구분): "
                ).split()
                search_by_category_and_intent(client, category, intent)
            elif choice == "6":
                print("\n--- 하이브리드 검색을 실행합니다 ---")
                try:
                    query = input("검색어 입력: ")
                    alpha_str = input("alpha 값 입력 (e.g., 0.4): ")
                    threshold_str = input("임계값(threshold) 입력 (e.g., 0.85): ")

                    alpha_float = float(alpha_str)
                    threshold_float = float(threshold_str)

                    search_with_hybrid(client, query, alpha_float, threshold_float)
                except ValueError:
                    print("❗️ alpha와 threshold는 숫자로 입력해야 합니다.")
                except Exception as e:
                    print(f"❗️ 처리 중 오류 발생: {e}")

            elif choice == "7":
                print("\n--- 하이브리드 검색(+카테고리)을 실행합니다 ---")
                try:
                    query = input("검색어 입력: ")
                    category = input("카테고리 입력: ")
                    alpha_str = input("alpha 값 입력 (e.g., 0.4): ")
                    threshold_str = input("임계값(threshold) 입력 (e.g., 0.85): ")

                    alpha_float = float(alpha_str)
                    threshold_float = float(threshold_str)

                    search_with_hybrid_and_category(
                        client, query, category, alpha_float, threshold_float
                    )
                except ValueError:
                    print("❗️ alpha와 threshold는 숫자로 입력해야 합니다.")
                except Exception as e:
                    print(f"❗️ 처리 중 오류 발생: {e}")
            elif choice == "8":
                diagnose_category_issue(client)
            elif choice == "9":
                delete_by_uuid(client)
            elif choice == "10":
                print("\n--- 속성(Property)으로 데이터 삭제 ---")
                prop_name = input(
                    "기준이 될 속성 이름을 입력하세요 (예: category 또는 messages): "
                ).strip()
                prop_value = input(f"'{prop_name}'의 값을 입력하세요: ").strip()

                if not prop_name or not prop_value:
                    print("❗️ 속성 이름과 값을 모두 입력해야 합니다.")
                else:
                    # 1단계: 먼저 검색해서 확인
                    targets = search_for_deletion(client, prop_name, prop_value)

                    # 2단계: 삭제 대상이 있으면 삭제 여부 묻고 진행
                    if targets:
                        delete_by_property(client, prop_name, prop_value)
            elif choice == "11":
                print(
                    "\n--- 두 컬렉션(ClassifyIntent, IntentList)의 검색 점수를 비교합니다 ---"
                )
                try:
                    query = input("비교할 검색어 입력: ")
                    category = input("비교할 카테고리 입력: ")
                    alpha_str = input("하이브리드 검색 alpha 값 입력 (e.g., 0.4): ")

                    alpha_float = float(alpha_str)

                    compare_search_scores(
                        client, embedding_model, query, alpha_float, category
                    )

                except ValueError:
                    print("❗️ alpha는 숫자로, 입력값은 형식에 맞게 입력해야 합니다.")
                except Exception as e:
                    print(f"❗️ 처리 중 오류 발생: {e}")
            elif choice == "12":
                check_collection_schemas(client)
            elif choice == "13":
                compare_production_logic(client, embedding_model)
            elif choice == "14":
                print("\n--- [TermDef] 하이브리드 검색을 실행합니다 ---")
                if not kiwi_analyzer:
                    print("❗️ Kiwipie 분석기를 로드할 수 없어 실행이 불가능합니다.")
                    continue
                try:
                    query = input("검색어 입력: ")
                    alpha_str = input("alpha 값 입력 (e.g., 0.4): ")
                    threshold_str = input("임계값(threshold) 입력 (e.g., 0.85): ")

                    alpha_float = float(alpha_str)
                    threshold_float = float(threshold_str)

                    search_termdef_hybrid(
                        client,
                        embedding_model,
                        kiwi_analyzer,
                        query,
                        alpha_float,
                        threshold_float,
                    )
                except ValueError:
                    print("❗️ alpha와 threshold는 숫자로 입력해야 합니다.")
                except Exception as e:
                    print(f"❗️ 처리 중 오류 발생: {e}")
            elif choice.lower() == "q":
                print("프로그램을 종료 합니다.")
                break
            else:
                print("잘못된 선택입니다. 다시 입력해주세요.")

        except Exception as e:
            print(f"❌ 메인 실행 중 예측하지 못한 오류 발생: {e}")

    if client and client.is_connected():
        client.close()
        print("\n🔗 Weaviate 클라이언트 연결을 모두 마치고 닫았습니다.")
