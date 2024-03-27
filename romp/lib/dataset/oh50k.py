import sys, os
from dataset.image_base import *
from dataset.base import Base_Classes, Test_Funcs
import json

default_mode = args().image_loading_mode

def OH50K(base_class=default_mode):
    class OH50K(Base_Classes[base_class]):
        def __init__(self,train_flag = False, split='test', regress_smpl=True, **kwargs):
            super(OH50K, self).__init__(train_flag, regress_smpl)
            self.data_folder = os.path.join(self.data_folder,'3DOH50K/')
            if not os.path.isdir(self.data_folder):
                self.data_folder = '/home/yusun/data_drive/dataset/3DOH50K/imageFiles'
            # self.data3d_dir = os.path.join(self.data_folder,'sequenceFiles')
            self.image_dir = os.path.join(self.data_folder,'images')
            # self.mode = mode
            self.split = split

            # self.regress_smpl = regress_smpl
            # self.val_sample_ratio = 5
            # self.scale_range = [1.1,2.]
            # self.dataset_name = {'PC':'pw3d_pc', 'NC':'pw3d_nc','OC':'pw3d_oc','vibe':'pw3d_vibe', 'normal':'pw3d_normal'}[mode]
            # self.use_org_annot_modes = ['normal','PC']

            if self.regress_smpl:
                self.smplr = SMPLR(use_gender=True)
                # self.root_inds = None

            logging.info('Start loading 3DOH50K data.')
            self.joint_mapper = constants.joint_mapping(constants.OH50K_24,constants.SMPL_ALL_54)
            self.joint3d_mapper = constants.joint_mapping(constants.OH50K_24,constants.SMPL_ALL_54)
            self.root_inds = [constants.SMPL_ALL_54['Pelvis']]

            # if self.regress_smpl:
            #     logging.info('loading SMPL regressor for mesh vertex calculation.')
            #     self.smplr = SMPLR(use_gender=True)
            

            annots_file_path = os.path.join(self.data_folder, 'annots.json')

            self.annots = json.load(open('/home/dcvl/MK/ROMP/dataset/3DOH50K/annots.json'))

            self.file_paths = list(self.annots.keys())
            # self.file_paths = [key for key in self.annots]
            self.kps_vis = (self.joint_mapper!=-1).astype(np.float32)[:,None]


        def get_image_info(self, index):
            img_name = self.file_paths[index]
            subject_id = 'S1'
            imgpath = os.path.join(self.image_dir, img_name +'.jpg')
            # print(imgpath)
            image = cv2.imread(imgpath)[:,:,::-1]

            info = self.annots[img_name].copy()

            # print(self.joint_mapper)
            kp2d = self.map_kps(np.array(info['smpl_joints_2d']), maps=self.joint_mapper)
            kp2ds = np.concatenate([kp2d,self.kps_vis], 1)[None]

            # print("#########################",np.array(info['smpl_joints_3d']).shape)    # (24, 3)


            # kp3ds = self.map_kps(np.array(info['smpl_joints_3d']), maps=self.joint3d_mapper)[None]
            # root_trans = kp3ds[:,self.root_inds].mean(1)
            # kp3ds -= root_trans[:,None]

            # root_rotation = np.array(info['cam'])[smpl_randidx]
            pose = np.array(info['pose']).squeeze(0)
            beta = np.array(info['betas']).squeeze(0)
            # print(pose[:66].shape, beta.shape)
            params = np.concatenate([pose[:66], beta], 0)[None]
            # print(params.shape)

            # root_trans = np.array(info['smpl_joints_3d'])[0].mean(0)[None]
            # print("################ 1", np.array(info['smpl_joints_3d']).shape) # (24, 3)
            # print("################ 2", np.array(info['smpl_joints_3d'])[0].shape)  # 3
            # print("################ 3", root_trans.shape)  # 1

            # world_coord = np.concatenate((np.array(info['smpl_joints_3d'])[0], [1]), axis=0)
            # extrinsic = np.array(info['extri'])
            # # print("################ 4", world_coord.shape, extrinsic.shape) # (2), (4, 4)
            # root_trans = np.matmul(extrinsic, world_coord)

            root_trans = np.array(info['smpl_joints_3d'])[0]

            verts = None
            if self.regress_smpl:
                gender = 'n'
                verts, kp3ds = self.smplr(pose, beta, gender)
            else:
                # world = np.array(info['smpl_joints_3d'])                    # 24, 3
                # intri = np.array(info['intri'])                             # 3, 3
                # extri = np.array(info['extri'])                             # 4, 4
                # trans = np.array(info['trans'])                             # 1, 3
                # scale = np.array(info['scale'])                             # 1

                # campose = extri.copy()
                # campose[:3, 3] = trans                             # 4, 4
                # inv_campose = np.linalg.inv(campose)                        # 4, 4
                # world_homo = np.hstack((world, np.ones((world.shape[0], 1))))     # 24, 4
                # camera_homo = np.dot(inv_campose, world_homo.T).T           # 24, 4
                # camera = camera_homo[:, :3] / camera_homo[:, 3:4]           # 24, 3
                # print(camera/scale[0])
                # # camera = np.dot(intri, camera.T).T                          # 24, 3
                # kp3ds = self.map_kps(camera/scale[0], maps=self.joint3d_mapper)[None]
                
                # mat = np.hstack((extri[:3, :3], trans.transpose()))         # 3, 4
                # camMats = np.dot(intri, mat)                                # 3, 4
                # world = np.concatenate((world, np.ones((1, 24))), axis=1)   # 24, 4
                # world = world.transpose()                                   # 4, 24
                # camera = np.dot(camMats, world)                             # 3, 24



                camkp3d = np.array(info['smpl_joints_3d'].copy())+np.array(info['trans'].copy())
                # camkp3d -= root_trans
                kp3ds = self.map_kps(camkp3d, maps=self.joint3d_mapper)[None]/info['scale'][0]
                # print(camkp3d)
            # kp3ds -= root_trans

            # R, T = self.annots[img_name]['extri']
            # camMats = np.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])
            camMats = np.array(info['intri'])

            # vmask_2d | 0: kp2d/bbox | 1: track ids | 2: detect all people in image
            # vmask_3d | 0: kp3d | 1: smpl global orient | 2: smpl body pose | 3: smpl body shape | 4: smpl verts | 5: depth
            img_info = {'imgpath': imgpath, 'image': image, 'kp2ds': kp2ds, 'track_ids': [1],\
                    'vmask_2d': np.array([[True,True,True]]), 'vmask_3d': np.array([[True,True,True,True,True,True]]),\
                    'kp3ds': kp3ds, 'params': params, 'root_trans': root_trans, 'verts': verts,\
                    'camMats': camMats, 'img_size': image.shape[:2], 'ds': 'oh50k'}

            if 'relative' in base_class:
                img_info['depth'] = np.array([[0, 0, 0, 0]])
            
            return img_info

    return OH50K


if __name__ == '__main__':
    dataset= OH50K(base_class=default_mode)(train_flag=False, regress_smpl=False)
    Test_Funcs[default_mode](dataset,with_3d=True,with_smpl=False)
    print('Done')