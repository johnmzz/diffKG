from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "http://localhost:9890/sparql"

def describe_entity(mid):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    query = f"""
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT DISTINCT ?p ?o
    FROM <http://freebase.com>
    WHERE {{
        ns:{mid} ?p ?o .
    }}
    LIMIT 200
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        if not bindings:
            print(f"🔍 No triples found for entity: {mid}")
            return

        print(f"\n📌 Triples for entity: ns:{mid}")
        for b in bindings:
            pred = b['p']['value'].replace("http://rdf.freebase.com/ns/", "")
            obj = b['o']
            if obj['type'] == 'uri' and obj['value'].startswith("http://rdf.freebase.com/ns/"):
                obj_value = obj['value'].replace("http://rdf.freebase.com/ns/", "")
            else:
                obj_value = obj['value']
            print(f"  ns:{mid} → {pred} → {obj_value}")

    except Exception as e:
        print(f"❌ SPARQL query failed: {e}")

if __name__ == "__main__":
    # 🔁 修改为你要查询的 MID
    describe_entity("m.07fj_")
