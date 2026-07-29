from neo4j import GraphDatabase
import os
import uuid
from dotenv import load_dotenv

load_dotenv()


class Neo4jService:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(
                os.getenv("NEO4J_USERNAME"),
                os.getenv("NEO4J_PASSWORD")
            )
        )

    def close(self):
        self.driver.close()

    def execute(self, query, params=None):
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    
    # Person
    

    def create_person(self, name, email):
        query = """
        MERGE (p:Person {email:$email})
        SET
            p.id=$id,
            p.name=$name
        RETURN p
        """

        return self.execute(query, {
            "id": str(uuid.uuid4()),
            "name": name,
            "email": email
        })

    
    # Technology
    

    def create_technology(self, name):
        query = """
        MERGE (t:Technology {name:$name})
        RETURN t
        """

        return self.execute(query, {
            "name": name
        })

    
    # Decision
    

    def create_decision(self, title, reason, confidence):

        decision_id = str(uuid.uuid4())

        query = """
        CREATE (d:Decision{
            id:$id,
            title:$title,
            reason:$reason,
            confidence:$confidence,
            created_at:datetime()
        })
        RETURN d
        """

        self.execute(query, {
            "id": decision_id,
            "title": title,
            "reason": reason,
            "confidence": confidence
        })

        return decision_id

    
    # Relationships
    

    def approve_decision(self, email, decision_id):
        query = """
        MATCH (p:Person {email:$email})
        MATCH (d:Decision {id:$decision_id})

        MERGE (p)-[:APPROVED]->(d)

        RETURN p,d
        """

        return self.execute(query, {
            "email": email,
            "decision_id": decision_id
        })

    def use_technology(self, decision_id, technology):
        query = """
        MATCH (d:Decision {id:$decision_id})
        MATCH (t:Technology {name:$technology})

        MERGE (d)-[:USES]->(t)

        RETURN d,t
        """

        return self.execute(query, {
            "decision_id": decision_id,
            "technology": technology
        })

    def supersede_decision(self, new_id, old_id):
        query = """
        MATCH (new:Decision {id:$new})
        MATCH (old:Decision {id:$old})

        MERGE (new)-[:SUPERSEDES]->(old)

        RETURN new,old
        """

        return self.execute(query, {
            "new": new_id,
            "old": old_id
        })

    def add_conflict(self, decision_a, decision_b):
        query = """
        MATCH (a:Decision {id:$a})
        MATCH (b:Decision {id:$b})

        MERGE (a)-[:CONFLICTS_WITH]->(b)

        RETURN a,b
        """

        return self.execute(query, {
            "a": decision_a,
            "b": decision_b
        })

    
    # Queries
    

    def get_decisions(self):
        query = """
        MATCH (d:Decision)
        RETURN d
        ORDER BY d.created_at DESC
        """

        return self.execute(query)

    def get_timeline(self):
        query = """
        MATCH (d:Decision)

        RETURN
        d.title,
        d.reason,
        d.created_at

        ORDER BY d.created_at
        """

        return self.execute(query)

    def get_conflicts(self):
        query = """
        MATCH (a:Decision)-[:CONFLICTS_WITH]->(b:Decision)

        RETURN
        a.title AS decision_a,
        b.title AS decision_b
        """

        return self.execute(query)

    def why_using(self, technology):
        query = """
        MATCH (d:Decision)-[:USES]->(t:Technology{name:$technology})
        OPTIONAL MATCH (p:Person)-[:APPROVED]->(d)

        RETURN
            d.title AS decision,
            d.reason AS reason,
            d.confidence AS confidence,
            collect(p.name) AS approvers
        """

        return self.execute(query, {
            "technology": technology
        })



# Example


if __name__ == "__main__":

    graph = Neo4jService()

    graph.create_person(
        "Alice",
        "alice@company.com"
    )

    graph.create_technology(
        "Kafka"
    )

    decision = graph.create_decision(
        "Use Kafka",
        "High throughput",
        0.94
    )

    graph.approve_decision(
        "alice@company.com",
        decision
    )

    graph.use_technology(
        decision,
        "Kafka"
    )

    print(graph.why_using("Kafka"))

    graph.close()