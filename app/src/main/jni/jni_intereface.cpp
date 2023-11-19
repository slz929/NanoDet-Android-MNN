// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

// #include<platform.h>

#include "NanoDet.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON


static NanoDet* g_nanodet = 0;
// static ncnn::Mutex lock;
static std::string img_truck= ""; 

class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    // nanodet
    {
        //  ncnn::MutexLockGuard g(lock);

        if (g_nanodet)
        {
            int width= rgb.cols;
            int height= rgb.rows;
            auto objects= g_nanodet->detect(img_truck, rgb, (unsigned char *)nullptr, rgb.cols, rgb.rows, 0.4, 0.45);

            g_nanodet->draw(rgb, objects);

        }
//         else
//         {
//             draw_unsupported(rgb);
//         }
    }

    // draw_fps(rgb);
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        // ncnn::MutexLockGuard g(lock);

        delete g_nanodet;
        g_nanodet = 0;
    }

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_nanodetmnn_nanodetMNN_loadModel(JNIEnv* env, jobject thiz, jstring name, jint modelid, jint cpugpu)
{
    if (modelid < 0 || modelid > 6 || cpugpu < 0 || cpugpu > 1)
    {
        return JNI_FALSE;
    }

    // AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
//    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    const char* modeltypes[] =
    {
        "nanodet_320"
    };

    const char* modeltype = modeltypes[(int)modelid];
    bool use_gpu = (int)cpugpu == 1;

    // reload
    {
        //  ncnn::MutexLockGuard g(lock);

        // if (use_gpu)
        // {
        //     // no gpu
        //     delete g_nanodet;
        //     g_nanodet = 0;
        // }
        // else
        {
            if (!g_nanodet){
                char model_name[256];
                const char *pathTemp = env->GetStringUTFChars(name, 0);
                std::string modelPath_base = pathTemp;
                std::string modelPath = pathTemp;
                 sprintf(model_name, "%s.mnn", modeltype);
                 std::string modelName= model_name;
                 modelPath= modelPath_base+ modelName;
                 img_truck=  modelPath_base+"truck.jpg";
                g_nanodet = new NanoDet(modelPath, use_gpu);
            }
        }
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_tencent_nanodetmnn_nanodetMNN_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int)facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_tencent_nanodetmnn_nanodetMNN_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_tencent_nanodetmnn_nanodetMNN_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

}
