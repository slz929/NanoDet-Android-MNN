#include "NanoDet.h"
#include "omp.h"

bool NanoDet::hasGPU = true;
bool NanoDet::toUseGPU = false;
NanoDet *NanoDet::detector = nullptr;

NanoDet::NanoDet(const std::string &mnn_path, bool useGPU) {
    toUseGPU = hasGPU && useGPU;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn mnn path", "%s", mnn_path.c_str());

    NanoDet_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    MNN::ScheduleConfig config;
    config.numThread = 4;
    if (useGPU) {
        config.type = MNN_FORWARD_OPENCL;
    }
    config.backupType = MNN_FORWARD_CPU;
    MNN::BackendConfig backendConfig;
    backendConfig.memory = MNN::BackendConfig::Memory_Normal;  // 内存
    backendConfig.power = MNN::BackendConfig::Power_Normal;  // 功耗
    backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_Normal;  // 精度
    config.backendConfig = &backendConfig;

    NanoDet_session = NanoDet_interpreter->createSession(config);
    input_tensor = NanoDet_interpreter->getSessionInput(NanoDet_session, nullptr);
}

NanoDet::~NanoDet() {
    NanoDet_interpreter->releaseModel();
    NanoDet_interpreter->releaseSession(NanoDet_session);
}

std::vector<BoxInfo>
NanoDet::detect(std::string& img_truck, cv::Mat &raw_image, unsigned char *image_bytes, int width, int height, double threshold, double nms_threshold) {
    std::vector<BoxInfo> result_list;
    if (raw_image.empty()) {
        LOGD("image is empty, please check!");
        return result_list;
    }

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "img_truck %s", img_truck.c_str());
    // raw_image= cv::imread(img_truck);
    // width= raw_image.cols;
    // height= raw_image.rows;

    auto dims = input_tensor->shape();
    MNN::Tensor::DimensionType dimensionType = input_tensor->getDimensionType();
    if (dimensionType == MNN::Tensor::DimensionType::TENSORFLOW) {
        in_n = dims[0];
        in_h = dims[1];
        in_w = dims[2];
        in_c = dims[3];
    } else if (dimensionType == MNN::Tensor::DimensionType::CAFFE) {
        in_n = dims[0];
        in_c = dims[1];
        in_h = dims[2];
        in_w = dims[3];
    } else {
        LOGW("other dimension type");
    }

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn mnn size", "%d %d %d %d %d %d %d %d", dims[0], dims[1],dims[2],dims[3],height,width, in_h,in_w);

    NanoDet_interpreter->resizeTensor(input_tensor, dims);
    NanoDet_interpreter->resizeSession(NanoDet_session);

    cv::Mat image;
    cv::resize(raw_image, image, cv::Size(in_w, in_h));

    MNN::CV::ImageProcess::Config config;
    std::memcpy(config.mean, mean_vals, sizeof(mean_vals));
    std::memcpy(config.normal, norm_vals, sizeof(norm_vals));
    config.filterType = MNN::CV::NEAREST;
    config.sourceFormat = MNN::CV::ImageFormat::RGB;
    config.destFormat = MNN::CV::ImageFormat::BGR;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(config));
    pretreat->convert(image.data, in_w, in_h, image.step[0], input_tensor);

    image.release();
    auto start = std::chrono::steady_clock::now();

    // run network
    NanoDet_interpreter->runSession(NanoDet_session);

    // get output data
    std::vector<std::vector<BoxInfo>> results;
    results.resize(num_class);

    for (const auto &head_info : heads_info) {
        MNN::Tensor *tensor_scores = NanoDet_interpreter->getSessionOutput(NanoDet_session, head_info.cls_layer.c_str());
        MNN::Tensor *tensor_boxes = NanoDet_interpreter->getSessionOutput(NanoDet_session, head_info.dis_layer.c_str());

        MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
        tensor_scores->copyToHostTensor(&tensor_scores_host);

        MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
        tensor_boxes->copyToHostTensor(&tensor_boxes_host);

        decode_infer(tensor_scores, tensor_boxes, head_info.stride, threshold, results);
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
//    LOGD("inference time:%fs", elapsed.count());

//    #pragma omp parallel for
    int cnt = 0;
    for (int i = 0; i < (int) results.size(); i++) {
        nms(results[i], nms_threshold);

        for (auto box : results[i]) {
            __android_log_print(ANDROID_LOG_DEBUG, "ncnn mnn box", "%d %f %d", cnt, box.score, box.label);
            box.x1 = box.x1 / in_w * width;
            box.x2 = box.x2 / in_w * width;
            box.y1 = box.y1 / in_h * height;
            box.y2 = box.y2 / in_h * height;
            result_list.push_back(box);
            cnt+=1;
        }
    }
//    LOGD("detect:%d objects", result_list.size());
    return result_list;
}

void NanoDet::decode_infer(MNN::Tensor *cls_pred, MNN::Tensor *dis_pred, int stride, float threshold,
                           std::vector<std::vector<BoxInfo>> &results) {
    int feature_h = in_h / stride;
    int feature_w = in_w / stride;

    //cv::Mat debug_heatmap = cv::Mat(feature_h, feature_w, CV_8UC3);
    for (int idx = 0; idx < feature_h * feature_w; idx++) {
        // scores is a tensor with shape [feature_h * feature_w, num_class]
        const float *scores = cls_pred->host<float>() + (idx * num_class);

        int row = idx / feature_w;
        int col = idx % feature_w;
        float score = 0;
        int cur_label = 0;
        for (int label = 0; label < num_class; label++) {
            if (scores[label] > score) {
                score = scores[label];
                cur_label = label;
            }
        }
        if (score > threshold) {
            // bbox is a tensor with shape [feature_h * feature_w, 4_points * 8_distribution_bite]
            const float *bbox_pred = dis_pred->host<float>() + (idx * 4 * (reg_max + 1));
            results[cur_label].push_back(disPred2Bbox(bbox_pred, cur_label, score, col, row, stride));
            //debug_heatmap.at<cv::Vec3b>(row, col)[0] = 255;
        }
    }
}

BoxInfo NanoDet::disPred2Bbox(const float *&dfl_det, int label, float score, int x, int y, int stride) {
    float ct_x = (x + 0.5f) * stride;
    float ct_y = (y + 0.5f) * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4);
    for (int i = 0; i < 4; i++) {
        float dis = 0;
        float *dis_after_sm = new float[reg_max + 1];
        activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
        for (int j = 0; j < reg_max + 1; j++) {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);
    float xmax = (std::min)(ct_x + dis_pred[2], (float) in_w);
    float ymax = (std::min)(ct_y + dis_pred[3], (float) in_h);

    return BoxInfo{xmin, ymin, xmax, ymax, score, label};
}

void NanoDet::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH) {
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH) {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            } else {
                j++;
            }
        }
    }
}


int NanoDet::draw(cv::Mat& rgb, const std::vector<BoxInfo>& objects)
{
    static const char* class_names[] = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
    };
    static const unsigned char colors[19][3] = {
        { 54,  67, 244},
        { 99,  30, 233},
        {176,  39, 156},
        {183,  58, 103},
        {181,  81,  63},
        {243, 150,  33},
        {244, 169,   3},
        {212, 188,   0},
        {136, 150,   0},
        { 80, 175,  76},
        { 74, 195, 139},
        { 57, 220, 205},
        { 59, 235, 255},
        {  7, 193, 255},
        {  0, 152, 255},
        { 34,  87, 255},
        { 72,  85, 121},
        {158, 158, 158},
        {139, 125,  96}
    };

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const BoxInfo& obj = objects[i];

        const unsigned char* color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb, cv::Point(obj.x1, obj.y1), cv::Point(obj.x2, obj.y2), cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.x1;
        int y = obj.y1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);

    }
    
    
    return 0;
}

inline float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

template<typename _Tp>
int activation_function_softmax(const _Tp *src, _Tp *dst, int length) {
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{0};

    for (int i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}
