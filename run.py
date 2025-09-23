import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from config import config
from utils import utils
from model.TS_AE import TS_AE
from model.KGMLC import KGMLC
from einops import rearrange
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score


def val_or_test_tsae(net, mode, dataloader, result_dict):
    with torch.no_grad():
        result_dict[mode] = 0.0
        for data in dataloader:
            imgs, flow, _, _ = data
            ori_imgs = rearrange(
                imgs[:, :-1, :, :],
                "b t c w h -> (b t) c w h",
            ).to(torch.device(config.device))
            next_imgs = rearrange(
                imgs[:, 1:, :, :],
                "b t c w h -> (b t) c w h",
            ).to(torch.device(config.device))
            flow = rearrange(flow, "b t c w h -> (b t) c w h").to(
                torch.device(config.device)
            )

            out = net(ori_imgs, flow)
            loss = F.l1_loss(out, next_imgs)
            result_dict[mode] += loss.item()
        result_dict[mode] = result_dict[mode] / len(dataloader)
    return result_dict


def val_or_test_kgmlc(net, mode, dataloader, result_dict):
    with torch.no_grad():
        result_dict[mode] = 0.0
        out_all = []
        label_all = []
        for data in dataloader:
            imgs, flow, label, _ = data
            ori_imgs = rearrange(
                imgs[:, :-1, :, :],
                "b t c w h -> (b t) c w h",
            ).to(torch.device(config.device))
            flow = rearrange(flow, "b t c w h -> (b t) c w h").to(
                torch.device(config.device)
            )
            label = label.to(torch.device(config.device))

            out = net(ori_imgs, flow)
            if out_all != []:
                for i, class_out in enumerate(out):
                    out_all[i] = torch.cat([out_all[i], class_out], dim=0)
            else:
                out_all = out
            label_all.append(label)

        label_all = rearrange(torch.cat(label_all), "b c-> c b")

        acc = []
        mac_f1 = []
        kappa = []
        # auc_s = []
        for i in range(len(out_all)):
            result_dict[mode] += F.nll_loss(out_all[i], label_all[i]).item()
            acc.append(out_all[i].argmax(dim=1).eq(label_all[i]).float().mean().item())
            mac_f1.append(
                f1_score(
                    out_all[i].argmax(dim=1).cpu().numpy(),
                    label_all[i].cpu().numpy(),
                    average="macro",
                )
            )
            kappa.append(
                cohen_kappa_score(
                    out_all[i].argmax(dim=1).cpu().numpy(), label_all[i].cpu().numpy()
                )
            )
            # auc_s.append(
            #     roc_auc_score(
            #         label_all[i].cpu().numpy(),
            #         out_all[i].cpu().numpy(),
            #         multi_class="ovo",
            #     )
            # )

        result_dict[mode] = result_dict[mode] / len(out_all)
        result_dict[mode.split("_")[0] + "_Accuracy"] = acc
        result_dict[mode.split("_")[0] + "_Macro F1"] = mac_f1
        result_dict[mode.split("_")[0] + "_Kappa"] = kappa
        # result_dict[mode.split("_")[0] + "_AUC"] = auc_s

        return result_dict


def train_tsae(net, trn_dataloader, val_dataloader, result_dict):
    opt = torch.optim.AdamW(
        net.parameters(), lr=config.pretrn_lr, weight_decay=config.pretrn_wd
    )
    earlyStopping = utils.EarlyStopping(patience=config.patience, delta=config.delta)

    ssim_loss = utils.SSIM()
    for i in range(config.pretrn_epoch):
        result_dict["loss"] = 0.0

        print(f"TSAE Epoch {i}/{config.pretrn_epoch}\n")
        pbar = tqdm(total=len(trn_dataloader))
        pbar.set_description_str(f"TSAE Epoch {i}/{config.pretrn_epoch}")
        for data in trn_dataloader:
            imgs, flow, _, _ = data
            ori_imgs = rearrange(
                imgs[:, :-1, :, :],
                "b t c w h -> (b t) c w h",
            ).to(torch.device(config.device))
            next_imgs = rearrange(
                imgs[:, 1:, :, :],
                "b t c w h -> (b t) c w h",
            ).to(torch.device(config.device))
            flow = rearrange(flow, "b t c w h -> (b t) c w h").to(
                torch.device(config.device)
            )

            opt.zero_grad()
            out = net(ori_imgs, flow)
            loss = -ssim_loss(out, next_imgs)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)

            result_dict["loss"] += loss.item()

        result_dict["loss"] = result_dict["loss"] / len(trn_dataloader)
        pbar.close()
        result_dict = val_or_test_tsae(net, "val_loss", val_dataloader, result_dict)
        earlyStopping(result_dict["val_loss"], _)
        if earlyStopping.earlyStop:
            result_dict["best_net"] = net
            print(f"Early stopping as epoch {i}")
            print(
                f"TSAE Epoch:{i}/{config.pretrn_epoch}: Train loss:{result_dict['loss']} Valid loss:{result_dict['val_loss']}"
            )

            break
        result_dict["best_net"] = net
        print(
            f"TSAE Epoch:{i}/{config.pretrn_epoch}: Train loss:{result_dict['loss']} Valid loss:{result_dict['val_loss']}"
        )

    return result_dict


def train_kgmlc(net, trn_dataloader, val_dataloader, result_dict):
    opt = torch.optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.wd)
    mtl_loss_adjust = utils.MultiLossLayer(3).to(torch.device(config.device))
    earlyStopping = utils.EarlyStopping(patience=config.patience, delta=config.delta)

    for i in range(config.epoch):
        result_dict["loss"] = 0.0

        print(f"KGMLC Epoch {i}/{config.epoch}\n")
        kpbar = tqdm(total=len(trn_dataloader))
        kpbar.set_description_str(f"KGMLC Epoch {i}/{config.epoch}")
        for data in trn_dataloader:
            imgs, flow, label, _ = data
            ori_imgs = rearrange(
                imgs[:, :-1, :, :].to(torch.device(config.device)),
                "b t c w h -> (b t) c w h",
            )
            flow = rearrange(
                flow.to(torch.device(config.device)), "b t c w h -> (b t) c w h"
            )
            label = label.to(torch.device(config.device))

            opt.zero_grad()
            outs = net(ori_imgs, flow)
            loss_set = None
            for j, out in enumerate(outs):
                loss_set = utils.ts_append(
                    loss_set,
                    F.nll_loss(out.float(), label[:, j].long()),
                )
            loss = mtl_loss_adjust.get_loss(loss_set)
            loss.backward()
            opt.step()

            kpbar.set_postfix(loss=loss.item())
            kpbar.update(1)

            result_dict["loss"] += loss.item()

        result_dict["loss"] = result_dict["loss"] / len(trn_dataloader)
        kpbar.close()

        result_dict = val_or_test_kgmlc(net, "val_loss", val_dataloader, result_dict)
        earlyStopping(result_dict["val_loss"], _)
        if earlyStopping.earlyStop:
            result_dict["best_net"] = net
            print(f"Early stopping as epoch {i}")
            print(
                f"KGMLC Epoch:{i}/{config.epoch}: Train loss:{result_dict['loss']} Valid loss:{result_dict['val_loss']} Valid acc:{result_dict['val_Accuracy']} Valid f1:{result_dict['val_Macro F1']} Valid kappa:{result_dict['val_Kappa']} "
            )
            break

        result_dict["best_net"] = net
        print(
            f"KGMLC Epoch:{i}/{config.epoch}: Train loss:{result_dict['loss']} Valid loss:{result_dict['val_loss']} Valid acc:{result_dict['val_Accuracy']} Valid f1:{result_dict['val_Macro F1']} Valid kappa:{result_dict['val_Kappa']} "
        )

    return result_dict


def run():
    exp = 0
    while os.path.exists("./log/exp_" + str(exp)):
        exp += 1
    os.makedirs("./log/exp_" + str(exp))
    for i in range(config.rep):
        trn_dataloader, val_dataloader, tst_dataloader = utils.get_data()
        os.makedirs(f"./log/exp_" + str(exp) + "/rep_" + str(i))

        result_dict = {
            "Path": "./log/exp_" + str(exp) + "/rep_" + str(i),
        }
        if config.pretrained == None and i == 0:
            result_dict["mode"] = "TS-AE"

            tsae_net = TS_AE(latent_dim=config.latent_dim).to(
                torch.device(config.device)
            )

            result_dict = train_tsae(
                tsae_net, trn_dataloader, val_dataloader, result_dict
            )
            utils.log_results(result_dict)
        else:
            result_dict["mode"] = "TS-AE"
            tsae_net = torch.load(config.pretrained)
            result_dict = val_or_test_tsae(
                tsae_net, "val_loss", val_dataloader, result_dict
            )
            utils.log_results(result_dict)

        result_dict["mode"] = "G-MLC"
        ks_mlc = KGMLC(tsae_net, "../Data/Zinc").to(torch.device(config.device))
        result_dict = train_kgmlc(ks_mlc, trn_dataloader, val_dataloader, result_dict)
        result_dict = val_or_test_kgmlc(
            ks_mlc, "test_loss", tst_dataloader, result_dict
        )
        utils.log_results(result_dict)


if __name__ == "__main__":
    run()
