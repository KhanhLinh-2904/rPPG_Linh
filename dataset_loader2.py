from dataset.MTTSDataset_Hao import MTTSDataset_Hao
from dataset.MTTSDataset_Hao_TSW import MTTSDataset_TSW

import h5py


def dataset_loader(train, save_root_path, model_name, dataset_name, window_length, fold = None, SW=None, ImgROI=None, is_test=False):
    if type(dataset_name) == list:
        train_file_paths = []  # Anh was here
        valid_file_paths = []
        
        if fold is not None:
            if train==0 or train==1:
                for i in dataset_name:
                    train_file_path = save_root_path + i + f"_train_{fold}.hdf5"
                    train_file_paths.append(train_file_path)
                    
                    valid_file_path = save_root_path + i + f"_test_{fold}.hdf5"
                    valid_file_paths.append(valid_file_path)
            else:
                test_file_path = save_root_path + dataset_name[0] + "_test_" + str(fold) + ".hdf5"     # 5fold test
                print(f"test_file_path={test_file_path}")
        else:
            for i in dataset_name:
                train_file_path = save_root_path + i + "_train.hdf5"
                train_file_paths.append(train_file_path)

                valid_file_path = save_root_path + i + "_test.hdf5"
                valid_file_paths.append(valid_file_path)

        train_files = []
        valid_files = []
        test_files = []

        for i in train_file_paths:
            train_file = h5py.File(i, 'r')
            train_files.append(train_file)
        for i in valid_file_paths:
            valid_file = h5py.File(i, 'r')
            valid_files.append(valid_file)


        # if you wanna training please comment these codes, or if wanna testing please uncomment it 
        if is_test:
            print("test_file_path: ", test_file_path)
            test_file = h5py.File(test_file_path, 'r')  
            test_files.append(test_file)
        # print("train_file", train_files)

  
    if model_name in ['MTTS', 'MTTS_CSTM', 'TSDAN']:
        # test_set = MTTSDataset(all_file, dataset_name, window_length, True)           
        if type(dataset_name) == list:
            if fold is None:
                test_set = MTTSDataset_Hao(valid_files, dataset_name, window_length, True)    #  overlapping_Hao non_FiveFold      
            else:
                print(f"test_sliding_window={SW}")
                test_set = MTTSDataset_TSW(test_files, dataset_name, window_length, True, sliding_windows=SW, ImgROI=ImgROI)  
                print("len: ", len(test_set))  # overlapping_Hao  FiveFold   
        else:
            test_set = MTTSDataset_Hao(test_file, dataset_name, window_length, True, ImgROI=ImgROI)   # nonoverlapping_Dao  FiveFold 
        
    else:
        raise Exception("Model name is not correct or model is not supported!")
    return test_set
