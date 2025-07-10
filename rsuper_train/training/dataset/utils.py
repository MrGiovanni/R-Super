import numpy as np

def get_dataset(args, mode, **kwargs):
    if args.pancreas_only or args.kidney_only:
        if args.dataset!='abdomenatlas_ufo':
            raise ValueError('Not Implemented: pancreas_only and kidney_only can only be used with abdomenatlas_ufo dataset')
    if args.UFO_only or args.Atlas_only:
        if args.dataset!='abdomenatlas_ufo':
            raise ValueError('Not Implemented: Atlas_only and UFO_only can only be used with abdomenatlas_ufo dataset')
    
    if args.dimension == '2d':
        if args.dataset == 'acdc':
            from .dim2.dataset_acdc import CMRDataset

            return CMRDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)

    else:
        if args.dataset == 'acdc':
            from .dim3.dataset_acdc import CMRDataset

            return CMRDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)
        elif args.dataset == 'lits':
            from .dim3.dataset_lits import LiverDataset

            return LiverDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)

        elif args.dataset == 'bcv':
            from .dim3.dataset_bcv import BCVDataset

            return BCVDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)

        elif args.dataset == 'kits':
            from .dim3.dataset_kits import KidneyDataset

            return KidneyDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)

        elif args.dataset == 'amos_ct':
            from .dim3.dataset_amos_ct import AMOSDataset

            return AMOSDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)

        elif args.dataset == 'amos_mr':
            from .dim3.dataset_amos_mr import AMOSDataset

            return AMOSDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)

        elif args.dataset == 'msd_lung':
            from .dim3.dataset_msd_lung import LungDataset

            return LungDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)

        elif args.dataset == 'abdomenatlas':
            from .dim3.dataset_abdomenatlas import AbdomenAtlasDataset
            
            return AbdomenAtlasDataset(args, mode=mode, seed=args.split_seed,
                                       all_train=args.all_train, crop_on_tumor=args.crop_on_tumor,
                                       load_augmented=args.load_augmented, save_destination=args.save_destination,
                                       save_augmented=args.save_augmented)
        
        elif args.dataset == 'abdomenatlas_ufo':
            from .dim3.dataset_abdomenatlas_UFO import AbdomenAtlasDataset
            
            if args.pancreas_only:
                tumor_classes=['pancreas']
            elif args.kidney_only:
                tumor_classes=['kidney']
            else:
                tumor_classes=None
            
            
            if tumor_classes is None:
                return AbdomenAtlasDataset(args, mode=mode, seed=args.split_seed,
                                        all_train=args.all_train, crop_on_tumor=args.crop_on_tumor,
                                        load_augmented=args.load_augmented, save_destination=args.save_destination,
                                        save_augmented=args.save_augmented, 
                                        Atlas_only=args.Atlas_only,UFO_only=args.UFO_only)
            else:
                return AbdomenAtlasDataset(args, mode=mode, seed=args.split_seed,
                                        all_train=args.all_train, crop_on_tumor=args.crop_on_tumor,
                                        load_augmented=args.load_augmented, save_destination=args.save_destination,
                                        save_augmented=args.save_augmented, tumor_classes=tumor_classes,
                                        Atlas_only=args.Atlas_only,UFO_only=args.UFO_only)
        else:
            raise ValueError("The specified dataset doesn't exist: %s" % args.dataset)
            
            



