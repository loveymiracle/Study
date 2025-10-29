import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import time
import os
import csv
import requests
import json
import warnings
import itertools
from opensearchpy import OpenSearch
from opensearch_dsl import Search
import opensearch_py_ml as oml
from opensearchpy.helers import bulk
from dotenv import load_dotenv
import gc
import sys
import base64
import argparse

warnings.filterwarnings("ignore")

# ===========================================================================
# =========================== input argument ================================

parser = argparse.ArgumentParser(description="baseym")
parser.add_argument("--baseym", type=str, help="baseym", required=True)
args = parser.parse_args()
load_dotenv()

print(f"baseym: {args.baseym}")
BASE_YM = args.baseym

YEAR = int(BASE_YM[:4])
MONTH = int(BASE_YM[4:6])

SIDO_CDs = [11, 26, 27, 28, 29, 30, 31, 36, 41, 43, 44, 46, 47, 48, 50, 51, 52]

# ===========================================================================
# =========================== input argument ================================
# ===========================================================================

opensearch_hosts = os.getenv("OPENSEARCH_HOSTS")
opensearch_username = os.getenv("OPENSEARCH_USERNAME")
opensearch_password = os.getenv("OPENSEARCH_PASSWORD")

host = [item.strip("' \n") for item in opensearch_hosts.split(",")]
auth = (opensearch_username, opensearch_password)

# ===========================================================================


class OpenSearchClient:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OpenSearchClient, cls).__new__(cls)
            cls._instance._initialize_client()
        return cls._instance

    def _initialize_client(self):
        try:
            opensearch_hosts
            if not opensearch_hosts:
                raise ValueError("OPENSEARCH_HOSTS 환경변수 없음")

            self.client = OpenSearch(
                hosts=host,
                http_auth=auth,
                use_ssl=False,
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=False,
                timeout=60,
            )
        except Exception as e:
            print(f"클라이언트 초기화 실패 : {e}")
            self.client = None

    @staticmethod
    def get_month_start_end(year_month):
        base_date = datetime.strptime(str(year_month), "%Y%m")
        start_date = base_date.replace(day=1)
        next_month = start_date.replace(day=28) + timedelta(days=4)
        last_date = next_month - timedelta(days=next_month.day)
        return start_date.strftime("%Y%m%d"), last_date.strftime("%Y%m%d")

    def get_client(self):
        if not self.client:
            print("재초기화 시도")
            self._initialize_client()
        return self.client

    def search_stay(self, index_name, day, BASE_YM):
        try:
            if index_name in [f"stay_sgg_day_{BASE_YM}"]:
                search = None
                search = {
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
            elif index_name in [f"stay_sgg_mons_{BASE_YM}"]:
                search = None
                search = {
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
            else:
                return None

            response = self.client.search(index=index_name, body=search)
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
            return None

    def _build_ratio(self, sido_cd, day, BASE_YM):
        min_cd = f"{sido_cd}000"
        max_cd = f"{sido_cd}999"
        if day is None:
            time_filter = {"term": {"BASE_YM": BASE_YM}}
        else:
            time_filter = {"term": {"BASE_YMD": day}}

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
            query_body = self._build_ratio(sido_cd, day, BASE_YM)
            msearch_body.append(query_body)

        try:
            response = self.client.msearch(body=msearch_body)
            results = {}
            for i, res in enumerate(response["responses"]):
                sido_cd = sido_cds[i]
                if res.get("aggregations"):
                    results[sido_cd] = {
                        "one": res["aggregations"]["one"]["tot"]["value"],
                        "two": res["aggregations"]["two"]["tot"]["value"],
                        "three": res["aggregations"]["three"]["tot"]["value"],
                    }
                else:
                    results[sido_cd] = {"one": 0, "two": 0, "three": 0}
            return results
        except Exception as e:
            return {}


if __name__ == "__main__":
    client_wrapper = OpenSearchClient()

    start, last = client_wrapper.get_month_start_end(BASE_YM)
    START_DAY = int(start)
    LAST_DAY = int(last)

    month_index_list = [f"stay_sgg_mons_{BASE_YM}", f"stay_ratio_mons_{YEAR}"]

    day_index_list = [
        f"stay_sgg_day_{BASE_YM}",
        f"stay_ratio_day_{YEAR}",
    ]

    print("@" * 30)
    print("일별 데이터 조회")
    print("@" * 30)
    for day in range(START_DAY, LAST_DAY + 1):
        stay_day_index = f"stay_sgg_day_{BASE_YM}"
        stay_day_results = client_wrapper.search_stay(stay_day_index, day, BASE_YM)

        for sido_cd, data in stay_day_results.items():
            print(f"{day} - SIDO_CD {sido_cd} - {stay_day_index}: one {data['one']}명")
            print(f"{day} - SIDO_CD {sido_cd} - {stay_day_index}: two {data['two']}명")
            print(
                f"{day} - SIDO_CD {sido_cd} - {stay_day_index}: three {data['three']}명"
            )

    for day in range(START_DAY, LAST_DAY + 1):
        ratio_day_index = f"stay_ratio_day_{YEAR}"
        ratio_day_results = client_wrapper.search_ratio(
            ratio_day_index, day, BASE_YM, YEAR, SIDO_CDs
        )

        for sido_cd, data in ratio_day_results.items():
            print(f"{day} - SIDO_CD {sido_cd} - {ratio_day_index}: one {data['one']}명")
            print(f"{day} - SIDO_CD {sido_cd} - {ratio_day_index}: two {data['two']}명")
            print(
                f"{day} - SIDO_CD {sido_cd} - {ratio_day_index}: three {data['three']}명"
            )
    print("=" * 30)
    print("월별 데이터 조회")
    print("=" * 30)

    stay_mons_index = f"stay_sgg_mons_{BASE_YM}"
    stay_mons_results = client_wrapper.search_stay(stay_mons_index, None, BASE_YM)

    for sido_cd, data in stay_mons_results.items():
        print(f"{BASE_YM} - SIDO_CD {sido_cd} - {stay_mons_index}: one {data['one']}명")
        print(f"{BASE_YM} - SIDO_CD {sido_cd} - {stay_mons_index}: two {data['two']}명")
        print(
            f"{BASE_YM} - SIDO_CD {sido_cd} - {stay_mons_index}: three {data['three']}명"
        )

    ratio_mons_index = f"stay_ratio_mons_{YEAR}"
    ratio_mons_results = client_wrapper.search_ratio(
        ratio_mons_index, None, BASE_YM, SIDO_CDs
    )
    for sido_cd, data in ratio_mons_results.items():
        print(
            f"{BASE_YM} - SIDO_CD {sido_cd} - {ratio_mons_index}: one {data['one']}명"
        )
        print(
            f"{BASE_YM} - SIDO_CD {sido_cd} - {ratio_mons_index}: two {data['two']}명"
        )
        print(
            f"{BASE_YM} - SIDO_CD {sido_cd} - {ratio_mons_index}: three {data['three']}명"
        )
