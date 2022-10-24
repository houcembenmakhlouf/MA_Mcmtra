import torch
import pickle
import numpy as np
import requests
import io
from torchvision import models
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from helpers import *

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Extract Image and CLIP Features")
    parser.add_argument(
        "--vtype",
        type=str,
        default="imagenet",
        help="imagenet | places | emotion | clip",
    )
    parser.add_argument("--gpu", type=int, default=2, help="0,1,..")
    parser.add_argument("--mvsa", type=str, default="single", help="single | multiple")
    parser.add_argument("--ht", type=bool, default=True, help="True | False")

    args = parser.parse_args()
    return args


def extract_img_feats(args):

    gpu = args.gpu
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    img_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dloc = "data/db_hbm/"

    with open(f"{dloc}prepared_data_w_dum_data_fr_topics.p", "rb") as f:
        loaded_obj = pickle.load(f)
    df = pd.DataFrame(loaded_obj)

    def get_resnet_feats():
        activation = {}
        debug = {"n_images": 0}

        def feature_hook(module, input, output):
            activation["avgpool"] = (
                output.view(-1, output.shape[1]).data.cpu().numpy().tolist()
            )

        if args.vtype == "imagenet":
            print("imgnet")
            model = models.__dict__["resnet50"](pretrained=True)
        elif args.vtype == "places":
            print("places")
            model_file = "pre_trained/resnet101_places_best.pth.tar"
            model = models.__dict__["resnet101"](pretrained=False, num_classes=365)
            checkpoint = torch.load(
                model_file, map_location=lambda storage, loc: storage
            )
            state_dict = {
                str.replace(k, "module.", ""): v
                for k, v in checkpoint["state_dict"].items()
            }
            model.load_state_dict(state_dict)
        elif args.vtype == "emotion":
            print("emotion")
            model_file = "pre_trained/best_emo_resnet50.pt"
            model = models.__dict__["resnet50"](pretrained=False, num_classes=8)
            model.load_state_dict(torch.load(model_file))

        model.eval().to(device)

        model._modules.get("avgpool").register_forward_hook(feature_hook)

        def get_img_embed_dict(image, model, device):
            image = img_transforms(image)
            img_inputs = image.to(device)
            img_inputs = torch.unsqueeze(img_inputs, dim=0)

            with torch.no_grad():
                outputs = model(img_inputs)

            feats = activation["avgpool"]
            logits = outputs.view(-1, outputs.shape[1]).data.cpu().numpy().tolist()

            embed_dict = {"feats": feats, "logits": logits}
            return embed_dict

        def get_image_from_url(url):
            if url is None:
                return None

            response = requests.get(url)

            image = None
            if response.status_code == 200:
                if response.headers["content-type"].split("/")[0] == "image":
                    image = Image.open(io.BytesIO(response.content)).convert("RGB")
                    debug["n_images"] += 1

            return image

        def get_img_embed(sample):

            sample_embed = {"tweet": None, "replies": [], "quotes": []}

            image = get_image_from_url(sample["tweet"]["image"])
            if image is not None:
                embed = get_img_embed_dict(image, model, device)
            else:
                embed = None

            sample_embed["tweet"] = embed

            for reply in sample["replies"]:
                image = get_image_from_url(reply["image"])
                if image is not None:
                    embed = get_img_embed_dict(image, model, device)
                else:
                    embed = None
                sample_embed["replies"].append(embed)

            for quote in sample["quotes"]:
                image = get_image_from_url(quote["image"])
                if image is not None:
                    embed = get_img_embed_dict(image, model, device)
                else:
                    embed = None
                sample_embed["quotes"].append(embed)

            return sample_embed

        tqdm.pandas()
        df["img_embed"] = df.progress_apply(get_img_embed, axis=1)

        img_embed = df["img_embed"].tolist()
        with open(f"features/{args.vtype}_img_embed_w_fr_topics.p", "wb") as f:
            pickle.dump(img_embed, f)

        print(debug["n_images"])

    if args.vtype != "clip":
        get_resnet_feats()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    extract_img_feats(args)

    # with open(f"features/imagenet_img_embed.p", "rb") as f:
    #     loaded_obj = pickle.load(f)
