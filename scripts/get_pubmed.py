from elasticsearch import Elasticsearch

# 连接 ES
es =  Elasticsearch(
            hosts="http://es-cn-5yd3k7npf0003ge8r.elasticsearch.aliyuncs.com:9200",
            basic_auth=("elastic", 'kYJpzt36QoFkPYBckHUnp'),
        )

# 关键词
keyword = "阿司匹林"

# 查询
response = es.search(
    index=["pubmed_first_index","pubmed_increment_first_index"],
    query={
        "match": {
            "title": keyword
        }
    }
)

# 输出结果
for hit in response["hits"]["hits"]:
    print(hit["_source"])