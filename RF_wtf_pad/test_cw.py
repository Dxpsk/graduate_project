import numpy as np
import torch
from models.RF import getRF
import torch.utils.data as Data
import const_rf as const
import csv
import pre_recall as pre_recall


def load_data(fpath):
    train = np.load(fpath, allow_pickle=True).item()
    train_X, train_y = train['dataset'], train['label']
    return train_X, train_y


def load_model(class_num, path, device):
    model = getRF(class_num)
    model.load_state_dict(torch.load(path))
    model = model.to(device)
    return model


if __name__ == '__main__':
    # TODO: change the test dataset path
    test_dataset = ['/data/Deep_fingerprint/processed_RF_data/wtf_pad_cw-test.npy']
    device = torch.device('cuda:5')
    # TODO: change the trained model path
    defense_model = load_model(const.num_classes, '/home/xjj/projects/graduate_project/RF_wtf_pad/model_cw', device).eval()

    for i, path in enumerate(test_dataset):
        features, test_y = load_data(path)

        test_x = torch.unsqueeze(torch.from_numpy(features), dim=1).type(torch.FloatTensor)
        test_x = test_x.to(device)
        test_y = torch.squeeze(torch.from_numpy(test_y)).type(torch.LongTensor)
        test_data = Data.TensorDataset(test_x, test_y)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
        website_res = []
        with torch.no_grad():
            for v, (x, y) in enumerate(test_loader):
                defense_output = defense_model(x).cpu().squeeze().detach().numpy()
                pre = np.argmax(defense_output)
                website_res.append([y.item(), pre])

        # You can find the test result in 'result/'
        cur_website_path = '/home/xjj/projects/graduate_project/RF_wtf_pad/result/wtf_pad_cw.csv'
        with open(cur_website_path, 'w+', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for item in website_res:
                writer.writerow(item)

        acc = pre_recall.pre_recCall(cur_website_path, '/home/xjj/projects/graduate_project/RF_wtf_pad/result/wtf_pad_cw_evaluate.csv', const.num_classes)
        print('avg acc:' + str(acc))
        #avg acc:0.9709473684210527