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

    def search_sido_aggs(self, index_name, day):
        try:
            search_body = None
            search_body = {
                "size": 0,
                "query": {"bool": {"filter": [{"term": {"BASE_YMD": day}}]}},
                "aggs": {
                    "sido": {
                        "terms": {
                            "field": "PSIDO",
                            "size": 100,
                            "order": {"_key": "asc"},
                        },
                        "aggs": {"tot": {"sum": {"field": "TOT"}}},
                    }
                },
            }

            response = self.client.search(index=index_name, body=search_body)
            results = {}

            for bucket in response["aggregations"]["sido"]["buckets"]:
                results[bucket["key"]] = bucket["tot"]["value"]
            return results
        except Exception as e:
            return {}

    def _build_range_query(self, sido_cd, day, query_type):
        if query_type == "sgg":
            min_cd = f"{sido_cd}000"
            max_cd = f"{sido_cd}999"
            field_type = {"range": {"PSGG": {"gte": min_cd, "lte": max_cd}}}
        elif query_type == "adm":
            min_cd = f"{sido_cd}000000"
            max_cd = f"{sido_cd}999999"
            field_type = {"range": {"PADM": {"gte": min_cd, "lte": max_cd}}}
        else:
            return None

        return {
            "size": 0,
            "query": {"bool": {"filter": [{"term": {"BASE_YMD": day}}, field_type]}},
            "aggs": {"tot": {"sum": {"field": "TOT"}}},
        }

    def search_range_msearch(self, index_name, day, sido_cds, query_type):
        msearch_body = []
        for sido_cd in sido_cds:
            msearch_body.append({"index": index_name})
            query_body = self._build_range_query(sido_cd, day, query_type)
            msearch_body.append(query_body)

        try:
            response = self.client.msearch(body=msearch_body)
            results = {}
            for i, res in enumerate(response["responses"]):
                sido_cd = sido_cds[i]
                if res.get("aggregations"):
                    results[sido_cd] = res["aggregations"]["tot"]["value"]
                else:
                    results[sido_cd] = 0
            return results
        except Exception as e:
            return {}

    def _build_term(self, sido_cd, day, BASE_YM, query_type, category):
        if query_type == "sgg":
            min_cd = f"{sido_cd}000"
            max_cd = f"{sido_cd}999"
            field_type = {"range": {"PDEPAR_SGG_CD": {"gte": min_cd, "lte": max_cd}}}
        elif query_type == "adm":
            min_cd = f"{sido_cd}000000"
            max_cd = f"{sido_cd}999999"
            field_type = {
                "range": {"PDEPAR_ADMNS_DONG_CD": {"gte": min_cd, "lte": max_cd}}
            }

        if category == "prps":
            agg_type = {
                "zero": {
                    "filter": {"term": {"MOV_PRPS_CD": 0}},
                    "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                },
                "one": {
                    "filter": {"term": {"MOV_PRPS_CD": 1}},
                    "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                },
                "two": {
                    "filter": {"term": {"MOV_PRPS_CD": 2}},
                    "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                },
                "three": {
                    "filter": {"term": {"MOV_PRPS_CD": 3}},
                    "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                },
                "four": {
                    "filter": {"term": {"MOV_PRPS_CD": 4}},
                    "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                },
                "five": {
                    "filter": {"term": {"MOV_PRPS_CD": 5}},
                    "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                },
                "six": {
                    "filter": {"term": {"MOV_PRPS_CD": 6}},
                    "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                },
            }
        elif category == "way":
            agg_type = {
                "zero": {
                    "filter": {"term": {"MOV_WAY_CD": 0}},
                    "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                },
                "one": {
                    "filter": {"term": {"MOV_WAY_CD": 1}},
                    "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                },
                "two": {
                    "filter": {"term": {"MOV_WAY_CD": 2}},
                    "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                },
                "three": {
                    "filter": {"term": {"MOV_WAY_CD": 3}},
                    "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                },
                "four": {
                    "filter": {"term": {"MOV_WAY_CD": 4}},
                    "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                },
                "five": {
                    "filter": {"term": {"MOV_WAY_CD": 5}},
                    "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                },
                "six": {
                    "filter": {"term": {"MOV_WAY_CD": 6}},
                    "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                },
                "seven": {
                    "filter": {"term": {"MOV_WAY_CD": 7}},
                    "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                },
            }

        return {"size": 0, "query": {"bool": {"filter": [{"term": {"BASE_YMD": day}}]}}}

    def search_term(self, index_name, day, BASE_YM, sido_cds, query_type, category):
        msearch_body = []
        for sido_cd in sido_cds:
            msearch_body.append({"index": index_name})
            query_body = self._build_term(sido_cd, day, BASE_YM, query_type, category)
            msearch_body.append(query_body)

        try:
            response = self.client.msearch(body=msearch_body)
            results = {}
            for i, res in enumerate(response["responses"]):
                if category == "prps":
                    if res.get("aggregations"):
                        results[sido_cd] = {
                            "zero": res["aggregations"]["zero"]["tot"]["value"],
                            "one": res["aggregations"]["one"]["tot"]["value"],
                            "two": res["aggregations"]["two"]["tot"]["value"],
                            "three": res["aggregations"]["three"]["tot"]["value"],
                            "four": res["aggregations"]["four"]["tot"]["value"],
                            "five": res["aggregations"]["five"]["tot"]["value"],
                            "six": res["aggregations"]["six"]["tot"]["value"],
                        }
                    else:
                        results[sido_cd] = {
                            "zero": 0,
                            "one": 0,
                            "two": 0,
                            "three": 0,
                            "four": 0,
                            "five": 0,
                            "six": 0,
                        }
                else:
                    if res.get("aggregations"):
                        results[sido_cd] = {
                            "zero": res["aggregations"]["zero"]["tot"]["value"],
                            "one": res["aggregations"]["one"]["tot"]["value"],
                            "two": res["aggregations"]["two"]["tot"]["value"],
                            "three": res["aggregations"]["three"]["tot"]["value"],
                            "four": res["aggregations"]["four"]["tot"]["value"],
                            "five": res["aggregations"]["five"]["tot"]["value"],
                            "six": res["aggregations"]["six"]["tot"]["value"],
                            "seven": res["aggregations"]["seven"]["tot"]["value"],
                        }
                    else:
                        results[sido_cd] = {
                            "zero": 0,
                            "one": 0,
                            "two": 0,
                            "three": 0,
                            "four": 0,
                            "five": 0,
                            "six": 0,
                            "seven": 0,
                        }
            return results
        except Exception as e:
            return {}

    def _build_time(self, sido_cd, day, BASE_YM, query_type):
        if query_type == "sgg":
            min_cd = f"{sido_cd}000"
            max_cd = f"{sido_cd}999"
            field_type = {"range": {"PDEPAR_SGG_CD": {"gte": min_cd, "lte": max_cd}}}
        elif query_type == "adm":
            min_cd = f"{sido_cd}000000"
            max_cd = f"{sido_cd}999999"
            field_type = {
                "range": {"PDEPAR_ADMNS_DONG_CD": {"gte": min_cd, "lte": max_cd}}
            }

        return {
            "size": 0,
            "query": {"bool": {"filter": [{"term": {"BASE_YMD": day}}, field_type]}},
            "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
        }

    def search_time(self, index_name, day, BASE_YM, sido_cds, query_type):
        msearch_body = []
        msearch_body.append({"index": index_name})
        query_body = self._build_time(sido_cd, day, BASE_YM, query_type)
        msearch_body.append(query_body)

        try:
            response = self.client.msearch(body=msearch_body)
            results = {}

            for i, res in enumerate(response["responses"]):
                sido_cd = sido_cds[i]
                if res.get("aggregations"):
                    results[sido_cd] = res["aggregations"]["tot"]["value"]
                else:
                    results[sido_cd] = 0
            return results
        except Exception as e:
            return {}

    def search_sido_ncell(self, index_name, day, ntype):
        if ntype == "pde":
            field_value = "PDEPAR_SIDO_CD"
        elif ntype == "det":
            field_value = "DETINA_SIDO_CD"
        else:
            field_value = "SIDO_CD"

        try:
            search_body = None
            search_body = {
                "size": 0,
                "query": {"bool": {"filter": [{"term": {"BASE_YMD": day}}]}},
                "aggs": {
                    "sido": {
                        "terms": {
                            "field": field_value,
                            "size": 100,
                            "order": {"_key": "asc"},
                        },
                        "aggs": {"tot": {"sum": {"field": "TOT_POPUL_NUM"}}},
                    }
                },
            }

            response = self.client.search(index=index_name, body=search_body)
            results = {}

            for bucket in response["aggregations"]["sido"]["buckets"]:
                results[bucket["key"]] = bucket["tot"]["value"]

        except Exception as e:
            return {}


if __name__ == "__main__":
    client_wrapper = OpenSearchClient()

    start, last = client_wrapper.get_month_start_end(BASE_YM)
    START_DAY = int(start)
    LAST_DAY = int(last)

    prps_adm_day_index = f"native_prps_admdong_day_{BASE_YM}"
    prps_sgg_day_index = f"native_prps_sgg_day_{BASE_YM}"
    prps_sido_day_index = f"native_prps_sido_day_sum_{YEAR}"

    way_adm_day_index = f"native_way_admdong_day_{BASE_YM}"
    way_sgg_day_index = f"native_way_sgg_day_{BASE_YM}"
    way_sido_day_index = f"native_way_sido_day_sum_{YEAR}"

    prps_term_adm_day_index = f"native_prps_age_admdong_day_{BASE_YM}"
    prps_term_sgg_day_index = f"native_prps_age_sgg_day_{BASE_YM}"

    way_term_adm_day_index = f"native_way_age_admdong_day_{BASE_YM}"
    way_term_sgg_day_index = f"native_way_age_sgg_day_{BASE_YM}"

    prps_time_adm_day_index = f"native_prps_time_admdong_day_{BASE_YM}"
    prps_time_sgg_day_index = f"native_prps_time_sgg_day_{BASE_YM}"

    way_time_adm_day_index = f"native_way_time_admdong_day_{BASE_YM}"
    way_time_sgg_day_index = f"native_way_time_sgg_day_{BASE_YM}"

    prps_out_day_index = f"native_prps_out_time_ncell_day_{BASE_YM}"
    prps_in_day_index = f"native_prps_in_time_ncell_day_{BASE_YM}"

    way_out_day_index = f"native_way_out_time_ncell_day_{BASE_YM}"
    way_in_day_index = f"native_way_in_time_ncell_day_{BASE_YM}"

    print("@" * 30)
    print("일별 데이터 조회")
    print("@" * 30)

    for day in range(START_DAY, LAST_DAY + 1):

        sido_results = client_wrapper.search_sido_aggs(prps_sido_day_index, day)
        for sido_cd, tot in sido_results.items():
            print(f"{day} - SIDO_CD {sido_cd} - {prps_sido_day_index}: {tot}명")

        sgg_results = client_wrapper.search_range_msearch(
            prps_sgg_day_index, day, SIDO_CDs, "sgg"
        )
        for sido_cd, tot in sgg_results.items():
            print(f"{day} - SGG_CD {sido_cd}000 - {prps_sgg_day_index}: {tot}명")

        adm_results = client_wrapper.search_range_msearch(
            prps_adm_day_index, day, SIDO_CDs, "adm"
        )
        for sido_cd, tot in adm_results.items():
            print(f"{day} - ADM_CD {sido_cd}000000 - {prps_adm_day_index}: {tot}명")

        w_sido_results = client_wrapper.search_sido_aggs(way_sido_day_index, day)
        for sido_cd, tot in w_sido_results.items():
            print(f"{day} - SIDO_CD {sido_cd} - {way_sido_day_index}: {tot}명")

        w_sgg_results = client_wrapper.search_range_msearch(
            way_sgg_day_index, day, SIDO_CDs, "sgg"
        )
        for sido_cd, tot in w_sgg_results.items():
            print(f"{day} - SGG_CD {sido_cd}000 - {way_sgg_day_index}: {tot}명")

        w_adm_results = client_wrapper.search_range_msearch(
            way_adm_day_index, day, SIDO_CDs, "adm"
        )
        for sido_cd, tot in w_adm_results.items():
            print(f"{day} - ADM_CD {sido_cd}000000 - {way_adm_day_index}: {tot}명")

        pterm_sgg_results = client_wrapper.search_term(
            prps_term_sgg_day_index, day, BASE_YM, SIDO_CDs, "sgg"
        )
        for sido_cd, tot in pterm_sgg_results.items():
            print(
                f"{day} - SGG_CD {sido_cd}000 - {prps_term_sgg_day_index}: zero tot{['zero']}명"
            )
            print(
                f"{day} - SGG_CD {sido_cd}000 - {prps_term_sgg_day_index}: one tot{['one']}명"
            )
            print(
                f"{day} - SGG_CD {sido_cd}000 - {prps_term_sgg_day_index}: two tot{['two']}명"
            )
            print(
                f"{day} - SGG_CD {sido_cd}000 - {prps_term_sgg_day_index}: three tot{['three']}명"
            )
            print(
                f"{day} - SGG_CD {sido_cd}000 - {prps_term_sgg_day_index}: four tot{['four']}명"
            )
            print(
                f"{day} - SGG_CD {sido_cd}000 - {prps_term_sgg_day_index}: five tot{['five']}명"
            )
            print(
                f"{day} - SGG_CD {sido_cd}000 - {prps_term_sgg_day_index}: six tot{['six']}명"
            )

        pterm_adm_results = client_wrapper.search_term(
            prps_term_adm_day_index, day, BASE_YM, SIDO_CDs, "adm"
        )
        for sido_cd, tot in pterm_adm_results.items():
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {prps_term_adm_day_index}: zero tot{['zero']}명"
            )
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {prps_term_adm_day_index}: one tot{['one']}명"
            )
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {prps_term_adm_day_index}: two tot{['two']}명"
            )
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {prps_term_adm_day_index}: three tot{['three']}명"
            )
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {prps_term_adm_day_index}: four tot{['four']}명"
            )
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {prps_term_adm_day_index}: five tot{['five']}명"
            )
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {prps_term_adm_day_index}: six tot{['six']}명"
            )

        wterm_sgg_results = client_wrapper.search_term(
            way_term_sgg_day_index, day, BASE_YM, SIDO_CDs, "sgg"
        )
        for sido_cd, tot in wterm_sgg_results.items():
            print(
                f"{day} - SGG_CD {sido_cd}000 - {way_term_sgg_day_index}: zero tot{['zero']}명"
            )
            print(
                f"{day} - SGG_CD {sido_cd}000 - {way_term_sgg_day_index}: one tot{['one']}명"
            )
            print(
                f"{day} - SGG_CD {sido_cd}000 - {way_term_sgg_day_index}: two tot{['two']}명"
            )
            print(
                f"{day} - SGG_CD {sido_cd}000 - {way_term_sgg_day_index}: three tot{['three']}명"
            )
            print(
                f"{day} - SGG_CD {sido_cd}000 - {way_term_sgg_day_index}: four tot{['four']}명"
            )
            print(
                f"{day} - SGG_CD {sido_cd}000 - {way_term_sgg_day_index}: five tot{['five']}명"
            )
            print(
                f"{day} - SGG_CD {sido_cd}000 - {way_term_sgg_day_index}: six tot{['six']}명"
            )
            print(
                f"{day} - SGG_CD {sido_cd}000 - {way_term_sgg_day_index}: seven tot{['seven']}명"
            )

        wterm_adm_results = client_wrapper.search_term(
            way_term_adm_day_index, day, BASE_YM, SIDO_CDs, "adm"
        )
        for sido_cd, tot in wterm_adm_results.itmes():
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {way_term_adm_day_index}: zero tot{['zero']}명"
            )
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {way_term_adm_day_index}: one tot{['one']}명"
            )
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {way_term_adm_day_index}: two tot{['two']}명"
            )
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {way_term_adm_day_index}: three tot{['three']}명"
            )
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {way_term_adm_day_index}: four tot{['four']}명"
            )
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {way_term_adm_day_index}: five tot{['five']}명"
            )
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {way_term_adm_day_index}: six tot{['six']}명"
            )
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {way_term_adm_day_index}: seven tot{['seven']}명"
            )

        ptime_sgg_results = client_wrapper.search_time(
            prps_time_sgg_day_index, day, BASE_YM, SIDO_CDs, "sgg"
        )
        for sido_cd, tot in ptime_sgg_results.itmes():
            print(f"{day} - SGG_CD {sido_cd}000 - {prps_time_sgg_day_index}: {tot}명")

        ptime_adm_results = client_wrapper.search_time(
            prps_time_adm_day_index, day, BASE_YM, SIDO_CDs, "adm"
        )
        for sido_cd, tot in ptime_adm_results.itmes():
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {prps_time_adm_day_index}: {tot}명"
            )

        wtime_sgg_results = client_wrapper.search_time(
            way_time_sgg_day_index, day, BASE_YM, SIDO_CDs, "sgg"
        )
        for sido_cd, tot in wtime_sgg_results.itmes():
            print(f"{day} - SGG_CD {sido_cd}000 - {way_time_sgg_day_index}: {tot}명")

        wtime_adm_results = client_wrapper.search_time(
            way_time_adm_day_index, day, BASE_YM, SIDO_CDs, "adm"
        )
        for sido_cd, tot in wtime_adm_results.itmes():
            print(
                f"{day} - ADM_CD {sido_cd}000000 - {prps_time_adm_day_index}: {tot}명"
            )

        pout_results = client_wrapper.search_sido_ncell(prps_out_day_index, day, "pde")
        for sido_cd, tot in pout_results.itmes():
            print(f"{day} - SIDO_CD {sido_cd} - {prps_out_day_index}: {tot}명")

        pin_results = client_wrapper.search_sido_ncell(prps_in_day_index, day, "det")
        for sido_cd, tot in pin_results.itmes():
            print(f"{day} - SIDO_CD {sido_cd} - {prps_in_day_index}: {tot}명")

        wout_results = client_wrapper.search_sido_ncell(way_out_day_index, day, "pde")
        for sido_cd, tot in wout_results.itmes():
            print(f"{day} - SIDO_CD {sido_cd} - {way_out_day_index}: {tot}명")

        win_results = client_wrapper.search_sido_ncell(way_in_day_index, day, "det")
        for sido_cd, tot in win_results.itmes():
            print(f"{day} - SIDO_CD {sido_cd} - {way_in_day_index}: {tot}명")
