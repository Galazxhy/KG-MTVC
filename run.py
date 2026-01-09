import os
import torch
import torch.nn.functional as F

from tqdm import tqdm
from config import config
from utils import utils
from model import Model, TS_AE
from einops import rearrange
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score
from torch.optim import lr_scheduler


def val_or_test_tsae(net, mode, dataloader, result_dict):
    with torch.no_grad():
        ssim_loss = utils.SSIM()
        result_dict[mode] = 0.0
        if config.model == "KG_MTVC":
            result_dict[mode + "_kl_loss"] = 0.0
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
            if config.model == "KG_MTVC":
                out, spa_mu, spa_logvar, tem_mu, tem_logvar = net(ori_imgs, flow)
                kl_loss_spa = net.kl_divergence(spa_mu, spa_logvar)
                kl_loss_tem = net.kl_divergence(tem_mu, tem_logvar)
                kl_loss = kl_loss_spa + kl_loss_tem

                loss = 1 - ssim_loss(out, next_imgs)
                result_dict[mode + "_kl_loss"] += kl_loss.item()
            else:
                out = net(ori_imgs, flow)
                loss = 1 - ssim_loss(out, next_imgs)
            result_dict[mode] += loss.item()
        result_dict[mode] = result_dict[mode] / len(dataloader)
        if config.model == "KG_MTVC":
            result_dict[mode + "_kl_loss"] = result_dict[mode + "_kl_loss"] / len(
                dataloader
            )
    return result_dict


def val_or_test_model(net, mode, dataloader, result_dict):
    with torch.no_grad():
        result_dict[mode] = 0.0
        out_all = []
        label_all = []
        mtl_loss_adjust = result_dict["multi_loss"]
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

            if config.model == "KG_MTVC":
                out = net(ori_imgs, flow)
            else:
                out = net(ori_imgs)
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
        loss_set = None

        for i in range(len(out_all)):
            if config.data_name == "HMDB":
                index = torch.where(label_all[i] != 0)
                loss_set = utils.ts_append(
                    loss_set,
                    F.nll_loss(out_all[i][index].float(), label_all[i][index].long()),
                )
                acc.append(
                    out_all[i][index]
                    .argmax(dim=1)
                    .eq(label_all[i][index])
                    .float()
                    .mean()
                    .item()
                )
                mac_f1.append(
                    f1_score(
                        out_all[i][index].argmax(dim=1).cpu().numpy(),
                        label_all[i][index].cpu().numpy(),
                        average="macro",
                    )
                )
                kappa.append(
                    cohen_kappa_score(
                        out_all[i][index].argmax(dim=1).cpu().numpy(),
                        label_all[i][index].cpu().numpy(),
                    )
                )
            else:
                loss_set = utils.ts_append(
                    loss_set, F.nll_loss(out_all[i].float(), label_all[i].long())
                )
                acc.append(
                    out_all[i].argmax(dim=1).eq(label_all[i]).float().mean().item()
                )
                mac_f1.append(
                    f1_score(
                        out_all[i].argmax(dim=1).cpu().numpy(),
                        label_all[i].cpu().numpy(),
                        average="macro",
                    )
                )
                kappa.append(
                    cohen_kappa_score(
                        out_all[i].argmax(dim=1).cpu().numpy(),
                        label_all[i].cpu().numpy(),
                    )
                )

        result_dict[mode] = loss_set.mean()
        result_dict[mode.split("_")[0] + "_Accuracy"] = acc
        result_dict[mode.split("_")[0] + "_Macro F1"] = mac_f1
        result_dict[mode.split("_")[0] + "_Kappa"] = kappa
        return result_dict


def train_tsae(net, trn_dataloader, val_dataloader, result_dict):
    opt = torch.optim.Adam(
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
            )
            next_imgs = rearrange(
                imgs[:, 1:, :, :],
                "b t c w h -> (b t) c w h",
            )
            flow = rearrange(flow, "b t c w h -> (b t) c w h")
            ori_imgs, flow, next_imgs = utils.transform_trn(ori_imgs, flow, next_imgs)
            ori_imgs = ori_imgs.to(torch.device(config.device))
            flow = flow.to(torch.device(config.device))
            next_imgs = next_imgs.to(torch.device(config.device))

            opt.zero_grad()
            if config.model == "KG_MTVC":
                out, spa_mu, spa_logvar, tem_mu, tem_logvar = net(ori_imgs, flow)
                kl_loss_spa = net.kl_divergence(spa_mu, spa_logvar)
                kl_loss_tem = net.kl_divergence(tem_mu, tem_logvar)
                kl_loss = kl_loss_spa + kl_loss_tem

                loss = 1 - ssim_loss(out, next_imgs) + config.vae_beta * kl_loss
            else:
                out = net(ori_imgs, flow)
                loss = 1 - ssim_loss(out, next_imgs)
            # loss = F.mse_loss(out, next_imgs)

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

    return net, result_dict


def train_model(net, trn_dataloader, val_dataloader, result_dict):
    mtl_loss_adjust = utils.MultiLossLayer(config.MTL_classes, init_log_sigma=0.5).to(
        torch.device(config.device)
    )
    opt = torch.optim.AdamW(
        [
            {"params": net.parameters()},
            {"params": mtl_loss_adjust.parameters()},
        ],
        lr=config.lr,
        weight_decay=config.wd,
    )

    earlyStopping = utils.EarlyStopping(patience=config.patience, delta=config.delta)
    scheduler = lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

    for i in range(config.epoch):
        result_dict["loss"] = 0.0

        print(f"KG_MTVC Epoch {i}/{config.epoch}\n")
        kpbar = tqdm(total=len(trn_dataloader))
        kpbar.set_description_str(f"KG_MTVC Epoch {i}/{config.epoch}")
        for data in trn_dataloader:
            imgs, flow, label, _ = data
            ori_imgs = rearrange(
                imgs[:, :-1, :, :],
                "b t c w h -> (b t) c w h",
            )
            flow = rearrange(flow, "b t c w h -> (b t) c w h")
            ori_imgs, flow, _ = utils.transform_trn(ori_imgs, flow, None)
            ori_imgs = ori_imgs.to(torch.device(config.device))
            flow = flow.to(torch.device(config.device))
            label = label.to(torch.device(config.device))

            opt.zero_grad()
            if config.model == "KG_MTVC":
                outs = net(ori_imgs, flow)
            else:
                outs = net(ori_imgs)
            loss_set = None
            for j, out in enumerate(outs):
                if config.data_name == "HMDB":
                    index = torch.where(label[:, j] != 0)
                    if index[0].shape[0] == 0:
                        loss_set = utils.ts_append(
                            loss_set, torch.tensor(0.0).to(torch.device(config.device))
                        )
                        continue
                    loss_set = utils.ts_append(
                        loss_set,
                        F.nll_loss(out[index].float(), label[:, j][index].long()),
                    )
                else:
                    loss_set = utils.ts_append(
                        loss_set, F.nll_loss(out.float(), label[:, j].long())
                    )
            # loss = loss_set.mean()
            loss = mtl_loss_adjust(loss_set)
            result_dict["multi_loss"] = mtl_loss_adjust
            loss.backward()
            opt.step()
            kpbar.set_postfix(loss=loss.item())
            kpbar.update(1)

            result_dict["loss"] += loss.item()

        result_dict["loss"] = result_dict["loss"] / len(trn_dataloader)
        scheduler.step()
        kpbar.close()

        result_dict = val_or_test_model(net, "val_loss", val_dataloader, result_dict)
        earlyStopping(result_dict["val_loss"], _)
        if earlyStopping.earlyStop:
            result_dict["best_net"] = net
            print(f"Early stopping as epoch {i}")
            print(
                f"KG_MTVC Epoch:{i}/{config.epoch}: Train loss:{result_dict['loss']} Valid loss:{result_dict['val_loss']} Valid acc:{result_dict['val_Accuracy']} Valid f1:{result_dict['val_Macro F1']} Valid kappa:{result_dict['val_Kappa']} "
            )
            break

        result_dict["best_net"] = net
        print(
            f"KG_MTVC Epoch:{i}/{config.epoch}: Train loss:{result_dict['loss']} Valid loss:{result_dict['val_loss']} Valid acc:{result_dict['val_Accuracy']} Valid f1:{result_dict['val_Macro F1']} Valid kappa:{result_dict['val_Kappa']} "
        )

    return result_dict


def run():
    exp = 0
    while os.path.exists("./log/exp_" + config.model + "_" + str(exp)):
        exp += 1
    os.makedirs("./log/exp_" + config.model + "_" + str(exp))
    for i in range(config.rep):
        print(f"------------------- Training Repetition {i} --------------------")
        print(f"-------------- Loading {config.data_name} Dataset --------------")
        trn_dataloader, val_dataloader, tst_dataloader = utils.get_data()
        os.makedirs(f"./log/exp_" + config.model + "_" + str(exp) + "/rep_" + str(i))

        result_dict = {
            "Path": "./log/exp_" + config.model + "_" + str(exp) + "/rep_" + str(i),
        }

        result_dict["mode"] = "MLC"
        if config.model == "KG_MTVC":
            result_dict["model"] = "KG_MTVC"
            ts_ae_model = TS_AE.TS_AE(
                config.latent_dim,
            ).to(torch.device(config.device))
            ts_ae_model, _ = train_tsae(
                ts_ae_model, trn_dataloader, val_dataloader, result_dict
            )

            model = Model.KG_MTVC(ts_ae_model, config.data_root).to(
                torch.device(config.device)
            )
        elif config.model == "Resnet3D":
            result_dict["model"] = "Resnet3D"
            model = Model.Resnet3D(config.data_root).to(torch.device(config.device))
        elif config.model == "Resnet3D_CS":
            result_dict["model"] = "Resnet3D_CS"
            model = Model.ResNet3D_CrossStitch(config.data_root).to(
                torch.device(config.device)
            )
        elif config.model == "X3D":
            result_dict["model"] = "X3D"
            model = Model.X3D(config.data_root).to(torch.device(config.device))
        elif config.model == "X3D_CS":
            result_dict["model"] = "X3D_CS"
            model = Model.X3D_CrossStitch(config.data_root).to(
                torch.device(config.device)
            )
        result_dict = train_model(model, trn_dataloader, val_dataloader, result_dict)
        result_dict = val_or_test_model(model, "test_loss", tst_dataloader, result_dict)
        utils.log_results(result_dict)


if __name__ == "__main__":
    run()
