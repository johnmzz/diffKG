from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://localhost:9890/sparql"  # 按需修改

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
            return bindings
        else:
            print("⚠️ WARNING: No results returned.")
            return []
    except Exception as e:
        print("❌ ERROR: Query failed")
        print(e)
        return []

def main():
    print(f"🔍 Checking SPARQL endpoint: {SPARQL_ENDPOINT}")
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)

    # 查询 1: 简单的三元组测试
    test_query(
        sparql,
        "SELECT * WHERE { ?s ?p ?o } LIMIT 5",
        "Basic triple query (any data)"
    )

    # 查询 2: 查找 Barack Obama 的实体 ID（mid）
    bindings = test_query(
        sparql,
        """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?entity
        FROM <http://freebase.com>
        WHERE {
            ?entity ns:type.object.name "Barack Obama"@en .
        }
        LIMIT 1
        """,
        "Find MID for 'Barack Obama'"
    )

    if not bindings:
        print("❌ ERROR: Cannot proceed without a valid entity for Barack Obama.")
        return

    obama_uri = bindings[0]["entity"]["value"]
    print(f"🔎 Found Obama entity URI: {obama_uri}")

    # 查询 3: Barack Obama 的名字
    test_query(
        sparql,
        f"""
        SELECT ?name
        FROM <http://freebase.com>
        WHERE {{
            <{obama_uri}> <http://rdf.freebase.com/ns/type.object.name> ?name .
            FILTER (lang(?name) = "en")
        }}
        """,
        "Get English name of Barack Obama"
    )

    # 查询 4: 出边关系（Obama -> ？）
    test_query(
        sparql,
        f"""
        SELECT ?relation ?tailEntity
        FROM <http://freebase.com>
        WHERE {{
            <{obama_uri}> ?relation ?tailEntity .
        }}
        LIMIT 5
        """,
        "Outgoing edges from Barack Obama"
    )

    # 查询 5: 入边关系（？ -> Obama）
    test_query(
        sparql,
        f"""
        SELECT ?relation ?headEntity
        FROM <http://freebase.com>
        WHERE {{
            ?headEntity ?relation <{obama_uri}> .
        }}
        LIMIT 5
        """,
        "Incoming edges to Barack Obama"
    )

    check_entity_label(sparql, "m.02qkg8m", "Madam Satan")

def check_entity_label(sparql, mid, expected_label):
    print(f"\n🔍 Checking MID: {mid} for expected label: '{expected_label}'")
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT ?name
    FROM <http://freebase.com>
    WHERE {{
        ns:{mid} ns:type.object.name ?name .
        FILTER (lang(?name) = "en")
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        if bindings:
            name = bindings[0]["name"]["value"]
            print(f"✅ Found label: '{name}'")
            if name.lower() == expected_label.lower():
                print("✅✅ MID matches the expected label!")
            else:
                print("⚠️ MID exists but label differs.")
        else:
            print("❌ MID not found in KG.")
    except Exception as e:
        print("❌ Query failed")
        print(e)


if __name__ == "__main__":
    main()
