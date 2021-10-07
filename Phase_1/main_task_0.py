from sklearn.datasets import fetch_olivetti_faces
from Util.dao_util import DAOUtil

NUM_OF_VARIANTS = 10


def main():
    """
        Executes Task-0
    """
    print('Welcome to Task 0 Demo')
    """
        Fetching the Olivetti Faces from Sklearn Dataset
    """
    faces = fetch_olivetti_faces()
    face_images = faces.images
    face_labels = faces.target
    """
        Connection to MongoDB
    """
    dao_util = DAOUtil()
    """
        Deleting all the records before running task-0
    """
    dao_util.delete_records()

    suffix = 1
    for i in range(0, len(face_labels)):
        face_image = face_images[i]
        """
            Computing unique image_label for each image
        """
        face_label = str(face_labels[i]) + '_' + str(suffix)
        """
            when the suffix reaches NUM_OF_VARIANTS (10), initialize it to 0 for second set of images.
        """
        suffix = suffix % NUM_OF_VARIANTS
        suffix += 1
        """
            Saves every vector to the DB with insert query. Please note that ideally, we should insert all records
            in one go. I am inserting record one by one for demo purpose only.
        """
        db_image_id = dao_util.save_vector_to_db(face_label, face_image.tolist())
        print('Successfully Inserted: ', db_image_id)


main()
