import shutil, os

#Каталоги
data_dir = 'images'
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

#Часть для тестов
test_data_portion = 0.10
val_data_portion = 0.10
nb_images = 2145

#index
start_val_data_index = int(nb_images*(1-val_data_portion-test_data_portion))
start_test_data_index = int(nb_images*(1-test_data_portion))

print(start_test_data_index)
print(start_val_data_index)

def create_directory(dir_name):
    if(os.path.exists(dir_name)):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name, 'glasses'))
    os.makedirs(os.path.join(dir_name, 'no_glasses'))

def copy_images(start, end, source_dir, dest_dir):
    for i in range(start, end):
        shutil.copy2(
            os.path.join(source_dir, 'glasses//glasses ({}).jpg'.format(i)),
            os.path.join(dest_dir, 'glasses')
        )
        shutil.copy2(
            os.path.join(source_dir, 'no_glasses//no_glasses ({}).jpg'.format(i)),
            os.path.join(dest_dir, 'no_glasses')
        )

create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)

copy_images(1, start_val_data_index, data_dir, train_dir)
copy_images(start_val_data_index, start_test_data_index, data_dir, val_dir)
copy_images(start_test_data_index, nb_images+1, data_dir, test_dir)