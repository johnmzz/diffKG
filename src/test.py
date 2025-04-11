from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://localhost:9890/sparql"  # 根据你的实际端口配置

def test_query(sparql, query, description):
    print(f"\n🧪 Testing: {description}")
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            print(f"✅ SUCCESS: Returned {len(bindings)} result(s). Example:")
            print(bindings[0])
        else:
            print("⚠️ WARNING: No results returned.")
    except Exception as e:
        print("❌ ERROR: Query failed")
        print(e)

def main():
    print(f"🔍 Checking SPARQL endpoint: {SPARQL_ENDPOINT}")
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)

    # 查询 1: 返回任意三元组（默认图）
    test_query(
        sparql,
        "SELECT * WHERE { ?s ?p ?o } LIMIT 5",
        "Basic triple query (any data)"
    )

    # 查询 2: 查询实体名称（Barack Obama）
    test_query(
        sparql,
        """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?name
        FROM <http://freebase.com>
        WHERE {
            ns:m.02jx3 ns:type.object.name ?name .
            FILTER (lang(?name) = "en")
        }
        """,
        "Entity name (Barack Obama) in English"
    )

    # 查询 3: 查询尾实体（从 Obama 出发）
    test_query(
        sparql,
        """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?relation ?tailEntity
        FROM <http://freebase.com>
        WHERE {
            ns:m.02jx3 ?relation ?tailEntity .
        }
        LIMIT 5
        """,
        "Outgoing edges from Barack Obama (tail entities)"
    )

    # 查询 4: 查询头实体（连向 Obama）
    test_query(
        sparql,
        """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?relation ?headEntity
        FROM <http://freebase.com>
        WHERE {
            ?headEntity ?relation ns:m.02jx3 .
        }
        LIMIT 5
        """,
        "Incoming edges to Barack Obama (head entities)"
    )

if __name__ == "__main__":
    main()
