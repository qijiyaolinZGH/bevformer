#改动prepare_train_data
import copy

import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
import mmcv
from os import path as osp
from mmdet.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .nuscnes_eval import NuScenesEval_custom
from .get_lidar2img_rt import get_lidar2img_rt
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.parallel import DataContainer as DC
import random
import h5py


@DATASETS.register_module()
class CCDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, queue_length=4, bev_size=(200, 200), overlap_test=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        #print(index)
        if index>299:
            return None
        #选出之前的3张，随机干掉1张
        queue = []
        index_list = list(range(index-self.queue_length, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[1:])
        index_list.append(index)
        #选出之前的3张，随机干掉1张
        #asd=[[238,239,240],[108,109,110],[41,42,43],[142,143,144],[77,78,79],[200,201,202],[65,66,67],[12,13,14],[84,85,86],[99,100,101],[222,223,224]]
        asd=[[238,239,240],[41,42,43]]
        index_list=random.choice(asd)
        '''
        if index%2:
            #index_list=[41,42,43]
            index_list=[238,239,240]
        else:
            #index_list=[41,42,43]
            index_list=[108,109,110]
        '''
        
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            if self.filter_empty_gt and \
                    (example is None or ~(example['gt_labels_3d']._data != -1).any()):
                return None
            queue.append(example)
        '''
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(queue)
        '''
        '''
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(self.pre_pipeline)
        print(self.pipeline)
        '''
        '''
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(index)
        '''
        
        asd=self.union2one(queue)
        '''
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(asd.keys())
        print(asd['img'].size())
        print(asd['img'].dim())
        print(asd['gt_bboxes_3d'].dim)
        print(dir(asd['gt_bboxes_3d']))
        print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
        '''
        
        #替换图片
        img=[]
        for i in index_list:
            img.append(self.CC_get_data_info_img(i))
        img=np.array(img)
        
        '''
        z=random.randint(0,10)
        #print(z)
        np.save('/root/autodl-tmp/BEVFormer/train/img_{}.npy'.format(z),img)
        '''
        
        #print(img.shape)#3, 380, 676
        
        #增加density_maps
        density_maps=[]
        with h5py.File('/root/autodl-tmp/BEVFormer/CCdata/ground_plane/train/Street_groundplane_train_dmaps_10.h5',"r") as gt_dm:
            density_maps=gt_dm['density_maps'][index_list[-1]]
            density_maps=np.transpose(density_maps, (2,0,1))
        density_maps = (density_maps-np.min(density_maps))/(np.max(density_maps)-np.min(density_maps))
        
        #print(density_maps.shape)
        
        
        asd['img'] = DC(torch.from_numpy(img).float(), cpu_only=False, stack=True)
        
        asd['density_maps'] = DC(torch.from_numpy(density_maps), cpu_only=False, stack=True)
        #asdasdasd.data
        
        #替换相机参数
        '''
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(asd.keys())#dict_keys(['img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'density_maps'])
        print(type(asd['img_metas']))#<class 'mmcv.parallel.data_container.DataContainer'>
        print(len(asd['img_metas']))#3
        print(type(asd['img_metas'].data))#<class 'dict'>
        print(asd['img_metas'].data.keys())#dict_keys([0, 1, 2])
        print(type(asd['img_metas'].data[0]))#<class 'dict'>
        print(asd['img_metas'].data[0].keys())#dict_keys(['filename', 'ori_shape', 'img_shape', 'lidar2img', 'pad_shape', 'scale_factor', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'sample_idx', 'prev_idx', 'next_idx', 'pts_filename', 'scene_token', 'can_bus', 'prev_bev_exists'])
        for i in range(len(asd['img_metas'])):
            print(type(asd['img_metas'].data[i]['lidar2img']))#<class 'list'>
            print(type(asd['img_metas'].data[i]['img_shape']))#<class 'list'>
            print(len(asd['img_metas'].data[i]['lidar2img']))#6
            print(type(asd['img_metas'].data[i]['lidar2img'][0]))#<class 'numpy.ndarray'>
            print(len(asd['img_metas'].data[i]['img_shape']))#6
            print(type(asd['img_metas'].data[i]['img_shape'][0]))#<class 'tuple'>
            print(asd['img_metas'].data[i]['img_shape'][0])#(480, 800, 3)
        
        #print(asd['img_metas'])
        asdasd
        '''
        lidar2img=[]
        for i in range(1):
            lidar2img.append(get_lidar2img_rt("view1"))
            lidar2img.append(get_lidar2img_rt("view2"))
            lidar2img.append(get_lidar2img_rt("view3"))
        metas_map = {}
        for i in range(len(asd['img_metas'])):
            metas_map[i]=asd['img_metas'].data[i]
            '''
            print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            print(type(metas_map[i]['lidar2img']))
            print(len(metas_map[i]['lidar2img']))
            print(type(metas_map[i]['lidar2img'][0]))
            print(metas_map[i]['lidar2img'][0].shape)
            '''
            metas_map[i]['lidar2img']=lidar2img #6个旋转参数的数组组成的列表
            metas_map[i]['img_shape']=[(380, 676, 3),(380, 676, 3),(380, 676, 3)]
            '''
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            print(type(metas_map[i]['lidar2img']))
            print(len(metas_map[i]['lidar2img']))
            print(type(metas_map[i]['lidar2img'][0]))
            print(metas_map[i]['lidar2img'][0].shape)
            print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            '''
        asd['img_metas'] = DC(metas_map, cpu_only=True)
        #asdasdad
        
        
        '''
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(asd.keys())
        print(asd['img'].size())
        print(asd['img'].dim())
        asdasdasd.data
        '''
        '''
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(asd.keys())
        print(asd['img'].size())
        print(asd['img'].dim())
        asdasdasd.data
        '''
        '''
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(asd.keys())
        print(asd['img_metas'].dim)
        asdasdasd.data
        '''
        '''
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(len(queue))
        print(len(asd))
        asdasdasd.data
        '''
        return asd
    
    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        
        #替换图片
        img=torch.tensor(self.CC_get_data_info_img(index))
        example['img'][0]=DC(img.float(), cpu_only=False, stack=True)
        
        '''
        print(' ')
        print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
        print(img.data.size())
        print(type(img))
        print(type(example))
        print(example.keys())
        print(type(example['img']))
        print(type(example['img'][0]))
        print(len(example['img']))
        print(example['img'][0].data.size())
        #asdasd
        '''
        
        #增加density_maps
        density_maps=[]
        with h5py.File('/root/autodl-tmp/BEVFormer/CCdata/ground_plane/train/Street_groundplane_train_dmaps_10.h5',"r") as gt_dm:
            density_maps=gt_dm['density_maps'][index]
            density_maps=np.transpose(density_maps, (2,0,1))
        example['density_maps'] = torch.from_numpy(density_maps)
        
        '''
        print(' ')
        print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
        print(type(example['density_maps']))
        print(type(example['density_maps'].data))
        print(example['density_maps'].data.size())
        asdasdasdads
        '''
        
        return example


    def union2one(self, queue):
        '''
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(type(queue))
        print(len(queue))
        print('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
        '''
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        prev_scene_token = None
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['scene_token'] != prev_scene_token:
                #metas_map[i]['prev_bev_exists'] = False
                metas_map[i]['prev_bev_exists'] = True
                prev_scene_token = metas_map[i]['scene_token']
                prev_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                prev_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][-1] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                tmp_pos = copy.deepcopy(metas_map[i]['can_bus'][:3])
                tmp_angle = copy.deepcopy(metas_map[i]['can_bus'][-1])
                metas_map[i]['can_bus'][:3] -= prev_pos
                metas_map[i]['can_bus'][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)
        
        
        '''
        print('cccccccccccccccccccccccccccccccccccccccc')
        print(type(queue[-1]))
        print(queue[-1]['img'].size())
        print('dddddddddddddddddddddddddddddddddddddd')
        '''
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]
        '''
        print('eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee')
        print(type(queue))
        print(len(queue))
        print('ffffffffffffffffffffffffffffffffffffffff')
        '''
        
        
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        rotation = Quaternion(input_dict['ego2global_rotation'])
        translation = input_dict['ego2global_translation']
        can_bus = input_dict['can_bus']
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        return input_dict
    def CC_get_data_info_img(self, index):
        with h5py.File('/root/autodl-tmp/BEVFormer/CCdata/camera_view/train/Street_view1_dmap_10.h5',"r") as view_1:
            img_1=view_1['images'][index]
            img_1=np.transpose(img_1, (2,0,1))
            img_1=np.broadcast_to(img_1,(3,img_1.shape[1],img_1.shape[2]))
        with h5py.File('/root/autodl-tmp/BEVFormer/CCdata/camera_view/train/Street_view2_dmap_10.h5',"r") as view_2:
            img_2=view_2['images'][index]
            img_2=np.transpose(img_2, (2,0,1))
            img_2=np.broadcast_to(img_2,(3,img_2.shape[1],img_2.shape[2]))
        with h5py.File('/root/autodl-tmp/BEVFormer/CCdata/camera_view/train/Street_view3_dmap_10.h5',"r") as view_3:
            img_3=view_3['images'][index]
            img_3=np.transpose(img_3, (2,0,1))
            img_3=np.broadcast_to(img_3,(3,img_3.shape[1],img_3.shape[2]))
        return [img_1,img_2,img_3]

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            '''
            print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
            print(type(data))
            print(data.keys())
            asdasdadada.size()
            '''
            return data

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        self.nusc = NuScenes(version=self.version, dataroot=self.data_root,
                             verbose=True)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val
        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir
    
    
    
    
    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        '''
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(result_names)
        asdasdasda.asdasda
        '''
        
        
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        print(result_names)
        print(type(result_files))
        print(result_files)
        print(result_files.keys)
        asdasdasda.asdasda

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict