from pymongo import MongoClient

"""
Data Access Object (DAO) Util Class
"""


class DAOUtil:
    """
    Initializes the db connection and prints the server instance
    """

    def __init__(self):
        client = MongoClient("mongodb+srv://dhruv_agja:test123@cluster0.21lv7.mongodb.net/myFirstDatabase?retryWrites=true&w=majority", ssl_cert_reqs=ssl.CERT_NONE)
        db = client.mwdb
        server_status = db.command("serverStatus")
        print(server_status)
        self.image_collection = db.images

    def save_to_db(self, dictionary):
        """
        :param dictionary:
        :return: inserts the dictionary object into the db
        """
        return self.image_collection.insert_one(dictionary)

    def save_vector_to_db(self, image_label, image_matrix):
        """
        :param image_label:
        :param image_matrix:
        :return: saves the object to db
        """
        image_object = {'label': image_label, 'image_pixels': image_matrix}
        image_id = self.image_collection.insert_one(image_object)
        return image_id

    def delete_records(self):
        """
        Deletes all records from DB
        """
        self.image_collection.remove()

    def get_image_for_label(self, image_id):
        """
        :param image_id:
        :return: retrieves image object for an image id
        """
        return self.image_collection.find_one({'label': image_id})

    def get_feature_descriptors_by_subject_id(self, subject_id):
        regex = 'image-.*-' + subject_id + '-[0-9]+.png'
        query_object = {'label': {'$regex': regex}}
        return list(self.get_records(query_object))

    def get_feature_descriptors_by_type_id(self, type_id):
        regex = 'image-'+type_id+'-[0-9]+-[0-9]+.png'
        query_object = {'label': {'$regex': regex}}
        # print(query_object)
        return list(self.get_records(query_object))

    def get_records(self, query_object):
        """
        :param query_object:
        :return: queries the database based on query object (dictionary)
        """
        return self.image_collection.find(query_object).sort([('label', 1)])
