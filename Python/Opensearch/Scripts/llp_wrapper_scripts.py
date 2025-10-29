from opensearchpy import OpenSearch
from datetime import datetime, timedelta
import os


class OpenSearchClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenSearchClient, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        try:
            opensearch_hosts = os.getenv("OPENSEARCH_HOSTS")
            username = os.getenv("OPENSEARCH_USERNAME")
            password = os.getenv("OPENSEARCH_PASSWORD")

            if not opensearch_hosts:
                raise ValueError("OPENSEARCH_HOSTS 환경변수 없음")

            hosts = [item.strip("' \n") for item in opensearch_hosts.split(",")]
            auth = (username, password)

            self.client = OpenSearch(
                hosts=hosts,
                http_auth=auth,
                use_ssl=False,
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=False,
                timeout=60,
            )
            print("[INFO] OpenSearch 클라이언트 초기화 성공")

        except Exception as e:
            print(f"[ERROR] 클라이언트 초기화 실패: {e}")
            self.client = None

    def get_client(self):
        if not self.client:
            print("OpenSearch 클라이언트 재초기화 시도")
            self._initialize_client()
        return self.client


class OpenSearchWrapper:
    def __init__(self):
        self.client = OpenSearchClient().get_client()

    @staticmethod
    def get_month_start_end(year_month):
        base_date = datetime.strptime(str(year_month), "%Y%m")
        start_date = base_date.replace(day=1)
        next_month = start_date.replace(day=28) + timedelta(days=4)
        last_date = next_month - timedelta(days=next_month.day)
        return start_date.strftime("%Y%m%d"), last_date.strftime("%Y%m%d")

    def search_stay(self, index_name, day, BASE_YM):
        try:
            if "day" in index_name:
                query = {
                    "size": 0,
                    "query": {"bool": {"filter": [{"term": {"BASE_YMD": day}}]}},
                    "aggs": {
                        "sido": {
                            "terms": {
                                "field": "SIDO_CD",
                                "size": 100,
                                "order": {"_key": "asc"},
                            },
                            "aggs": {
                                "one": {
                                    "filter": {"term": {"STAY_TIME_CD": 1}},
                                    "aggs": {
                                        "tot": {"sum": {"field": "TOT_POPOUL_NUM"}}
                                    },
                                },
                                "two": {
                                    "filter": {"term": {"STAY_TIME_CD": 2}},
                                    "aggs": {
                                        "tot": {"sum": {"field": "TOT_POPOUL_NUM"}}
                                    },
                                },
                                "three": {
                                    "filter": {"term": {"STAY_TIME_CD": 3}},
                                    "aggs": {
                                        "tot": {"sum": {"field": "TOT_POPOUL_NUM"}}
                                    },
                                },
                            },
                        }
                    },
                }
            else:
                query = {
                    "size": 0,
                    "query": {"bool": {"filter": [{"term": {"BASE_YM": BASE_YM}}]}},
                    "aggs": {
                        "sido": {
                            "terms": {
                                "field": "SIDO_CD",
                                "size": 100,
                                "order": {"_key": "asc"},
                            },
                            "aggs": {
                                "one": {
                                    "filter": {"term": {"STAY_TIME_CD": 1}},
                                    "aggs": {
                                        "tot": {"sum": {"field": "TOT_POPUL_NUM"}}
                                    },
                                },
                                "two": {
                                    "filter": {"term": {"STAY_TIME_CD": 2}},
                                    "aggs": {
                                        "tot": {"sum": {"field": "TOT_POPUL_NUM"}}
                                    },
                                },
                                "three": {
                                    "filter": {"term": {"STAY_TIME_CD": 3}},
                                    "aggs": {
                                        "tot": {"sum": {"field": "TOT_POPUL_NUM"}}
                                    },
                                },
                            },
                        }
                    },
                }

            response = self.client.search(index=index_name, body=query)
            results = {}
            for bucket in response["aggregations"]["sido"]["buckets"]:
                sido_cd = bucket["key"]
                results[sido_cd] = {
                    "one": bucket["one"]["tot"]["value"],
                    "two": bucket["two"]["tot"]["value"],
                    "three": bucket["three"]["tot"]["value"],
                }
            return results

        except Exception as e:
            print(f"[ERROR] search_stay 실패: {e}")
            return {}

    def _build_ratio(self, sido_cd, day, BASE_YM):
        min_cd = f"{sido_cd}000"
        max_cd = f"{sido_cd}999"
        time_filter = (
            {"term": {"BASE_YMD": day}} if day else {"term": {"BASE_YM": BASE_YM}}
        )

        return {
            "size": 0,
            "query": {
                "bool": {
                    "filter": [
                        {"range": {"SGG_CD": {"gte": min_cd, "lte": max_cd}}},
                        time_filter,
                    ]
                }
            },
            "aggs": {
                "one": {
                    "filter": {"term": {"STAY_TIME_CD": 1}},
                    "aggs": {"tot": {"sum": {"field": "STAY_AVG"}}},
                },
                "two": {
                    "filter": {"term": {"STAY_TIME_CD": 2}},
                    "aggs": {"tot": {"sum": {"field": "STAY_AVG"}}},
                },
                "three": {
                    "filter": {"term": {"STAY_TIME_CD": 3}},
                    "aggs": {"tot": {"sum": {"field": "STAY_AVG"}}},
                },
            },
        }

    def search_ratio(self, index_name, day, BASE_YM, sido_cds):
        msearch_body = []
        for sido_cd in sido_cds:
            msearch_body.append({"index": index_name})
            msearch_body.append(self._build_ratio(sido_cd, day, BASE_YM))

        try:
            response = self.client.msearch(body=msearch_body)
            results = {}
            for i, res in enumerate(response["responses"]):
                sido_cd = sido_cds[i]
                if "aggregations" in res:
                    results[sido_cd] = {
                        "one": res["aggregations"]["one"]["tot"]["value"],
                        "two": res["aggregations"]["two"]["tot"]["value"],
                        "three": res["aggregations"]["three"]["tot"]["value"],
                    }
                else:
                    results[sido_cd] = {"one": 0, "two": 0, "three": 0}
            return results
        except Exception as e:
            print(f"[ERROR] msearch 실패: {e}")
            return {}


if __name__ == "__main__":
    BASE_YM = "202510"
    YEAR = int(BASE_YM[:4])
    SIDO_CDs = [11, 26, 27, 28, 29, 30, 31, 36, 41, 43, 44, 46, 47, 48, 50, 51, 52]

    wrapper = OpenSearchWrapper()

    start, last = wrapper.get_month_start_end(BASE_YM)
    print(f"조회 기간: {start} ~ {last}")

    results = wrapper.search_stay(f"stay_sgg_day_{BASE_YM}", "20251001", BASE_YM)
    print(json.dumps(results, ensure_ascii=False, indent=2))

    ratio_results = wrapper.search_ratio(
        f"stay_ratio_day_{YEAR}", "20251001", BASE_YM, SIDO_CDs
    )
    print(json.dumps(ratio_results, ensure_ascii=False, indent=2))
