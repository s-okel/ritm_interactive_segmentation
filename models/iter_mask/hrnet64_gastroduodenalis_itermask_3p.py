from isegm.utils.exp_imports.default import *
label = "gastroduodenalis"
MODEL_NAME = f'{label}_hrnet64_iter'


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)


def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (96, 96)
    model_cfg.num_max_points = 24

    model = HRNetModel(width=64, ocr_width=64, with_aux_output=True, use_leaky_relu=True,
                       use_rgb_conv=False, use_disks=True, norm_radius=1, with_prev_mask=True,
                       one_input_channel=False)

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W64)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 32 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1.0
    loss_cfg.instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.instance_aux_loss_weight = 0.4

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)),
        Flip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.03, scale_limit=0,
                         rotate_limit=(-3, 3), border_mode=0, p=0.75),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size),
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)

    trainset = PancDataset(
        split='train',
        label=label,
        one_input_channel=False,
        augmentator=train_augmentator,
        min_object_area=0,
        keep_background_prob=0.01,
        points_sampler=points_sampler,
    )

    valset = PancDataset(
        split='val',
        label=label,
        one_input_channel=False,
        augmentator=val_augmentator,
        min_object_area=0,
        points_sampler=points_sampler,
        epoch_len=500
    )

    optimizer_params = {
        'lr': 5e-4, 'betas': (0.9, 0.999), 'eps': 1e-8
    }

    lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           milestones=[200, 215], gamma=0.1)
    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 5), (100, 1)],
                        image_dump_interval=200,
                        metrics=[AdaptiveIoU(), DiceScore()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=3)
    trainer.run(num_epochs=220)
