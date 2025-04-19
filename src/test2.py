from SPARQLWrapper import SPARQLWrapper, JSON

def count_neighbors(mid: str, endpoint: str = "http://localhost:9890/sparql") -> int:
    sparql = SPARQLWrapper(endpoint)
    query = f"""
    define sql:big-data-const 0
    PREFIX ns: <http://rdf.freebase.com/ns/>
    SELECT (COUNT(DISTINCT ?neighbor) AS ?neighborCount)
    WHERE {{
      {{
        ns:{mid} ?relation ?neighbor .
      }}
      UNION
      {{
        ?neighbor ?relation ns:{mid} .
      }}
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        count = results["results"]["bindings"][0]["neighborCount"]["value"]
        return int(count)
    except Exception as e:
        print("Query failed:", e)
        return -1

# 示例调用
print(count_neighbors("m.0ftn8"))  # Jamaica
