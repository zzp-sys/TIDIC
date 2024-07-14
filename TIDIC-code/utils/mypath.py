
class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10', 'stl-10', 'cifar-20', 'imagenet_dog', 'imagenet_10'}
        assert(database in db_names)

        if database == 'cifar-10':
            return '/home/ubuntu/DeepClustering/dataset/cifar-10/'
        
        elif database == 'cifar-20':
            return '/home/ubuntu/DeepClustering/dataset/cifar-20'

        elif database == 'stl-10':
            return '/home/ubuntu/DeepClustering/dataset/stl-10/'
        
        elif database == 'imagenet_10':
            return '/home/ubuntu/DeepClustering/dataset/imagenet-10/'
        
        elif database == 'imagenet_dog':
            return '/home/ubuntu/DeepClustering/dataset/imagenet-dog/'
        
        else:
            raise NotImplementedError
