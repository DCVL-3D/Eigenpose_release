# import os
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender
import trimesh
import numpy as np
import torch

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P

class Renderer(object):

    def __init__(self, focal_length=443.4, height=512, width=512,**kwargs):
        # original: focal_length=600, 443.4, 548, 352
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.camera_center = np.array([width / 2., height / 2.])
        self.focal_length = focal_length
        self.colors = [
                        (.7, .7, .6, 1.),
                        (.7, .5, .5, 1.),  # Pink
                        (.5, .5, .7, 1.),  # Blue
                        (.5, .55, .3, 1.),  # capsule
                        (.3, .5, .55, 1.),  # Yellow
                    ]

    def __call__(self, verts, faces, colors=None,focal_length=None,camera_pose=None,org_res=None,cam=None,**kwargs):
        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])

        #self.renderer.viewport_height = img.shape[0]
        #self.renderer.viewport_width = img.shape[1]
        # print("verts#"*10, verts.shape)
        # print("faces#"*10, faces.shape)
        num_people = verts.shape[0]
        verts = verts.detach().cpu().numpy()
        if isinstance(faces, torch.Tensor):
            faces = faces.detach().cpu().numpy()

        # Create a scene for each image and render all meshes
        # scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
        scene = pyrender.Scene(bg_color=[1,1,1,1], ambient_light=(0.3, 0.3, 0.3))

        
        # Create camera. Camera will always be at [0,0,0]
        # CHECK If I need to swap x and y
        # print('1 camera pose 2 focal_length, focal_length', camera_pose, focal_length)
        if camera_pose is None:
            camera_pose = np.eye(4)
            # camera_translation = np.array([0,0,100])
            # camera_translation = np.array([0,0, 2*self.focal_length/512])
            # camera_pose[:3, 3] = np.eye(3) @ camera_translation
            # camera_pose[2,3] = 100.
            # camera_pose[2,2] = -0.0002
            # print("############## camera_pose: ", camera_pose)

        if focal_length is None:
            fx,fy = self.focal_length, self.focal_length
        else:
            fx,fy = focal_length, focal_length
        # print('fx,fy', fx,fy)

        h, w = org_res
        # print("################# h, w: ", h, w)

        # camera = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy, cx=self.camera_center[0], cy=self.camera_center[1])
        # camera = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy, cx=self.camera_center[0], cy=self.camera_center[1], znear=0.0000000001, zfar=100.0)
        # camera = pyrender.camera.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.0000000001, zfar=100.0)

        # Create light source
        # light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
        # for every person in the scene

        sx, tx, ty = cam
        sy = sx.clone()
        # print('0',sx.item(), sy.item())

        if h > w:
            tx /= sy
            ty /= sy
            # sy *= h/w
            sx *= h/w

        else:
            tx /= sx
            ty /= sx
            # sx *= w/h
            sy *= w/h

        # print('1',sx.item(), sy.item())

        camera = WeakPerspectiveCamera(scale=[sx, sy], translation=[tx, ty], znear=0., zfar=100.)
        scene.add(camera, pose=camera_pose)


        mesh = trimesh.Trimesh(verts, faces)
        mesh.apply_transform(rot)
        trans = np.array([0,0,0])
        mesh_color = np.array([1., .9, .75, 1.])
        # if colors is None:
        #     mesh_color = self.colors[0] #self.colors[n % len(self.colors)]
        # else:
        #     mesh_color = colors[n % len(colors)]
        # print("########", mesh_color.shape)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            roughnessFactor=0.6,
            alphaMode='OPAQUE',
            # baseColorFactor=np.array([0.7, 0.7, 0.6, 1]))
            baseColorFactor=mesh_color)
        mesh = pyrender.Mesh.from_trimesh(
            mesh,
            material=material)
        scene.add(mesh, 'mesh')

        # Use 3 directional lights
        light_pose = np.eye(4)
        light_pose[:3, 3] = np.array([0, -1, 1]) + trans
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([0, 1, 1]) + trans
        scene.add(light, pose=light_pose)
        light_pose[:3, 3] = np.array([1, 1, 2]) + trans
        scene.add(light, pose=light_pose)
        # Alpha channel was not working previously need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA|pyrender.RenderFlags.SHADOWS_SPOT)
        # scene.remove_node(cam_node)
        # color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA|pyrender.RenderFlags.SHADOWS_SPOT)
        return 1-color



    # def __call__(self, verts, faces, colors=None,focal_length=None,camera_pose=None,org_res=None,cam=None,**kwargs):
    #     # Need to flip x-axis
    #     rot = trimesh.transformations.rotation_matrix(
    #         np.radians(180), [1, 0, 0])

    #     #self.renderer.viewport_height = img.shape[0]
    #     #self.renderer.viewport_width = img.shape[1]
    #     # print("verts#"*10, verts.shape)
    #     # print("faces#"*10, faces.shape)
    #     num_people = verts.shape[0]
    #     verts = verts.detach().cpu().numpy()
    #     if isinstance(faces, torch.Tensor):
    #     	faces = faces.detach().cpu().numpy()

    #     # Create a scene for each image and render all meshes
    #     scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
    #                            ambient_light=(0.3, 0.3, 0.3))

        
    #     # Create camera. Camera will always be at [0,0,0]
    #     # CHECK If I need to swap x and y
    #     # print('1 camera pose 2 focal_length, focal_length', camera_pose, focal_length)
    #     if camera_pose is None:
    #         camera_pose = np.eye(4)
    #         # camera_translation = np.array([0,0,100])
    #         # camera_translation = np.array([0,0, 2*self.focal_length/512])
    #         # camera_pose[:3, 3] = np.eye(3) @ camera_translation
    #         # camera_pose[2,3] = 100.
    #         # camera_pose[2,2] = -0.0002
    #         # print("############## camera_pose: ", camera_pose)

    #     if focal_length is None:
    #         fx,fy = self.focal_length, self.focal_length
    #     else:
    #         fx,fy = focal_length, focal_length
    #     # print('fx,fy', fx,fy)

    #     h, w = org_res
    #     # print("################# h, w: ", h, w)

    #     # camera = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy, cx=self.camera_center[0], cy=self.camera_center[1])
    #     # camera = pyrender.camera.IntrinsicsCamera(fx=fx, fy=fy, cx=self.camera_center[0], cy=self.camera_center[1], znear=0.0000000001, zfar=100.0)
    #     # camera = pyrender.camera.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.0000000001, zfar=100.0)

    #     # Create light source
    #     light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.5)
    #     # for every person in the scene

    #     for n in range(num_people):

    #         sx, tx, ty = cam[n]
    #         sy = sx

    #         if h > w:
    #             tx /= sy
    #             ty /= sy
    #             sy *= h/w
    #         else:
    #             tx /= sx
    #             ty /= sx
    #             sx *= w/h

    #         camera = WeakPerspectiveCamera(scale=[sx, sy], translation=[tx, ty], znear=0., zfar=100.)
    #         scene.add(camera, pose=camera_pose)


    #         mesh = trimesh.Trimesh(verts[n], faces[n])
    #         mesh.apply_transform(rot)
    #         trans = np.array([0,0,0])
    #         if colors is None:
    #             mesh_color = self.colors[0] #self.colors[n % len(self.colors)]
    #         else:
    #             mesh_color = colors[n % len(colors)]
    #         # print("########", mesh_color.shape)
    #         material = pyrender.MetallicRoughnessMaterial(
    #             metallicFactor=0.0,
    #             alphaMode='OPAQUE',
    #             # baseColorFactor=np.array([0.7, 0.7, 0.6, 1]))
    #             baseColorFactor=np.array([1, 1, 0.7, 1]))
    #         mesh = pyrender.Mesh.from_trimesh(
    #             mesh,
    #             material=material)
    #         scene.add(mesh, 'mesh')

    #         # Use 3 directional lights
    #         light_pose = np.eye(4)
    #         light_pose[:3, 3] = np.array([0, -1, 1]) + trans
    #         scene.add(light, pose=light_pose)
    #         light_pose[:3, 3] = np.array([0, 1, 1]) + trans
    #         scene.add(light, pose=light_pose)
    #         light_pose[:3, 3] = np.array([1, 1, 2]) + trans
    #         scene.add(light, pose=light_pose)
    #     # Alpha channel was not working previously need to check again
    #     # Until this is fixed use hack with depth image to get the opacity
    #     color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    #     # scene.remove_node(cam_node)
    #     # color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA|pyrender.RenderFlags.SHADOWS_SPOT)
    #     return color

    def delete(self):
        self.renderer.delete()

def get_renderer(test=False,**kwargs):
    renderer = Renderer(**kwargs)
    if test:
        import cv2, pickle, os
        import torch
        from config import args
        model = pickle.load(open('/home/dcvl/MK/ROMP_Exp07_2/model_data/smpl_models/SMPL_NEUTRAL.pkl','rb'), encoding='latin1')
        np_v_template = torch.from_numpy(np.array(model['v_template'])).cuda().float()[None]
        face = model['f'].astype(np.int32)[None]
        np_v_template = np_v_template.repeat(2,1,1)
        np_v_template[1] += 0.3
        np_v_template[:,:,2] += 6
        result = renderer(np_v_template, face)
        cv2.imwrite('test_pyrenderer.png',(result[:,:,:3]*255).astype(np.uint8))
    return renderer

if __name__ == '__main__':
    get_renderer(test=True, perps=True)
