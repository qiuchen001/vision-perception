from pymilvus import MilvusClient


class MilvusClientWrapper:
    def __init__(self, uri, db_name):
        self.client = MilvusClient(uri=uri, db_name=db_name)

    def create_schema(self, auto_id=False, enable_dynamic_fields=True, description=""):
        return self.client.create_schema(auto_id=auto_id, enable_dynamic_fields=enable_dynamic_fields,
                                         description=description)

    def create_collection(self, collection_name, schema):
        return self.client.create_collection(collection_name=collection_name, schema=schema)

    def query(self, collection_name, expr):
        return self.client.query(collection_name, expr)

    def insert(self, collection_name, data):
        return self.client.insert(collection_name, data)

    def create_index(self, collection_name, field_name, index_type, metric_type, params):
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=field_name,
            index_type=index_type,
            metric_type=metric_type,
            params=params
        )
        return self.client.create_index(collection_name, index_params)
