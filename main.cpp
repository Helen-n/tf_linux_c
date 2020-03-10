//
// Created by sergio on 12/05/19.
//
#include "Model.h"
#include "Tensor.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iterator>

using namespace cv;
using namespace std;

Mat convert_rgb_to_y(const Mat& src_img)
{
    Mat dst_img(src_img.rows,src_img.cols,CV_32FC1);
	if (src_img.channels() == 1)
	{
		dst_img = src_img;
		return dst_img;
	}
	else
	{
		for (int i = 0; i < src_img.rows; i++)
		{
			for (int j = 0; j < src_img.cols; j++)
			{
				// [[65.738 / 256.0, 129.057 / 256.0, 25.064 / 256.0]]
				dst_img.at<float>(i,j) = int(src_img.at<Vec3b>(i, j)[0])*(25.064 / 256.0) + int(src_img.at<Vec3b>(i, j)[1])*(129.057 / 256.0) + int(src_img.at<Vec3b>(i, j)[2])*(65.738 / 256.0) + 16.0;
			}
		}
	}
	return dst_img;
}


Mat convert_rgb_to_ycbcr(const Mat& src_img)
{
    Mat dst_img(src_img.rows,src_img.cols,CV_32FC3);

    for (int i = 0; i < src_img.rows; i++)
    {
        for (int j = 0; j < src_img.cols; j++)
        {
				//[65.738  / 256.0, 129.057 / 256.0, 25.064 / 256.0]
				//[-37.945 / 256.0, -74.494 / 256.0, 112.439 / 256.0]
				//[112.439 / 256.0, -94.154 / 256.0, -18.285 / 256.0]
            dst_img.at<Vec3f>(i, j)[0] = int(src_img.at<Vec3b>(i, j)[0])*(25.064 / 256.0)  + int(src_img.at<Vec3b>(i, j)[1])*(129.057 / 256.0) + int(src_img.at<Vec3b>(i, j)[2])*(65.738 / 256.0) + 16.0;
            dst_img.at<Vec3f>(i, j)[1] = int(src_img.at<Vec3b>(i, j)[0])*(112.439 / 256.0) + int(src_img.at<Vec3b>(i, j)[1])*(-74.494 / 256.0) + int(src_img.at<Vec3b>(i, j)[2])*(-37.945 / 256.0) + 128.0;
            dst_img.at<Vec3f>(i, j)[2] = int(src_img.at<Vec3b>(i, j)[0])*(-18.285 / 256.0) + int(src_img.at<Vec3b>(i, j)[1])*(-94.154 / 256.0) + int(src_img.at<Vec3b>(i, j)[2])*(112.439 / 256.0) + 128.0;
        }
    }
	return dst_img;
}


Mat convert_ycbcr_to_rgb(const Mat& ycbcr_image)
{
    Mat rgb_img(ycbcr_image.rows,ycbcr_image.cols,CV_32FC3);
	for (int i = 0; i < ycbcr_image.rows; i++)
	{
		for (int j = 0; j < ycbcr_image.cols; j++)
		{
			float y_data = ycbcr_image.at<Vec3f>(i, j)[0] - 16.0;
			float cr_data = ycbcr_image.at<Vec3f>(i, j)[1] - 128.0;
			float cb_data = ycbcr_image.at<Vec3f>(i, j)[2] - 128.0;
			rgb_img.at<Vec3f>(i, j)[2] = y_data * (298.082 / 256.0) + cr_data * (0)                + cb_data * (408.583 / 256.0);
			rgb_img.at<Vec3f>(i, j)[1] = y_data * (298.082 / 256.0) + cr_data * (-100.291 / 256.0) + cb_data * (-208.120 / 256.0);
			rgb_img.at<Vec3f>(i, j)[0] = y_data * (298.082 / 256.0) + cr_data * (516.412 / 256.0)  + cb_data * (0);
		}
	}
	return rgb_img;
}

Mat convert_y_and_cbcr_to_rgb(const Mat& y_image, const Mat& cbcr_image)
{
	Mat merg_mat, split_crcb[3];
	vector<Mat>  split_ycrcb;
	split(cbcr_image, split_crcb);
	split_ycrcb.push_back(y_image);
	split_ycrcb.push_back(split_crcb[1]);
	split_ycrcb.push_back(split_crcb[2]);
	merge(split_ycrcb, merg_mat);
	return convert_ycbcr_to_rgb(merg_mat);
}

template<typename _Tp>
Mat convert_vector_to_mat(vector<_Tp> vec, int channels, int rows)
{
   Mat tmp_mat = Mat(vec);
   Mat dst = tmp_mat.reshape(channels,rows),clone();
   return dst;
}



int main()
{
    // Create model
    Model m("example_mnist.pb");
    m.restore("checkpoint/dcscn_ar_75_0.ckpt");

    // Create Tensors
    auto input_x = new Tensor(m, "x");
    auto input_y = new Tensor(m, "y");
    auto input_f = new Tensor(m, "input_fusion");
    auto input_d = new Tensor(m, "dropout_keep_rate");
    auto input_t = new Tensor(m, "is_training");
    auto prediction = new Tensor(m, "output");

    Mat img, cbcr_img, y_img;
      // Read image
    img = imread("test1.jpg");
    y_img = convert_rgb_to_y(img);
    cbcr_img = convert_rgb_to_ycbcr(img);
    // Put image in vector
    std::vector<float> img_data;
    img_data.assign(y_img.begin<float>(), y_img.end<float>());

    // Feed data to input tensor
    std::vector<std::int64_t> img_dims= {1,img.cols ,img.rows,1};
    cout << img.cols  <<endl;
    cout <<  img.rows << endl;
    const std::vector<std::int64_t> dims1 = {1};
    auto data_size1 = sizeof(float);
    for (auto i : dims1)
    {
       data_size1 *= i;
    }

    auto data1 = static_cast<float*>(std::malloc(data_size1));

    std::vector<float> vals1 = {1.0};

    std::copy(vals1.begin(), vals1.end(), data1); // init input_vals.

    const std::vector<std::int64_t> dims2 = {1};
    auto data_size2 = sizeof(int);
    for (auto i : dims2)
    {
       data_size2 *= i;
    }

    auto data2 = static_cast<int*>(std::malloc(data_size2));

    std::vector<int> vals2 = {0};

    std::copy(vals2.begin(), vals2.end(), data2); // init input_vals.


    // Feed data to input tensor
    input_x->set_data(img_data, img_dims);
    input_y->set_data(img_data, img_dims);
    input_f->set_data(img_data, img_dims);

    input_d->set_data(vals1,dims1);
    //input_t->set_data(vals2,dims2);
    // Run and show predictions
    m.run({input_x,input_y,input_f,input_d}, prediction);
    // Get tensor with predictions
    auto result = prediction->get_data<float>();
    Mat y_result = convert_vector_to_mat(result,1,img.rows);
    cout << y_result.cols  <<endl;
    cout <<  y_result.rows << endl;
    Mat result_img = convert_y_and_cbcr_to_rgb(y_result,cbcr_img);
    cout << result_img.channels()  <<endl;
    cout <<  result_img.rows << endl;
    cout << result_img.cols  <<endl;
    Mat show_img;
    result_img.convertTo(show_img,CV_8UC3);
    imshow("src_img",img);
    imshow("result_img",show_img);
    waitKey(0);
    // Maximum prob
    //for(int i=0; i<result.size(); i++)
    //{
     //   cout<<result[i]<<endl;
   // }



	return 0;
}

/*
#include "Model.h"
#include "Tensor.h"

#include <numeric>
#include <iomanip>

int main() {
    Model model("load_model.pb");
    model.init();

    auto input_a = new Tensor(model, "input_a");
    auto input_b = new Tensor(model, "input_b");
    auto output  = new Tensor(model, "result");

    const std::vector<std::int64_t> dims = {1, 5,1, 12};
    auto data_size = sizeof(float);
    for (auto i : dims)
    {
       data_size *= i;
    }

    auto data = static_cast<float*>(std::malloc(data_size));

    std::vector<float> vals = {
    -0.4809832f, -0.3770838f, 0.1743573f, 0.7720509f, -0.4064746f, 0.0116595f, 0.0051413f, 0.9135732f, 0.7197526f, -0.0400658f, 0.1180671f, -0.6829428f,
    -0.4810135f, -0.3772099f, 0.1745346f, 0.7719303f, -0.4066443f, 0.0114614f, 0.0051195f, 0.9135003f, 0.7196983f, -0.0400035f, 0.1178188f, -0.6830465f,
    -0.4809143f, -0.3773398f, 0.1746384f, 0.7719052f, -0.4067171f, 0.0111654f, 0.0054433f, 0.9134697f, 0.7192584f, -0.0399981f, 0.1177435f, -0.6835230f,
    -0.4808300f, -0.3774327f, 0.1748246f, 0.7718700f, -0.4070232f, 0.0109549f, 0.0059128f, 0.9133330f, 0.7188759f, -0.0398740f, 0.1181437f, -0.6838635f,
    -0.4807833f, -0.3775733f, 0.1748378f, 0.7718275f, -0.4073670f, 0.0107582f, 0.0062978f, 0.9131795f, 0.7187147f, -0.0394935f, 0.1184392f, -0.6840039f,
    };

    std::copy(vals.begin(), vals.end(), data); // init input_vals.




    input_a->set_data(vals,dims);
    input_b->set_data(vals,dims);

    model.run({input_a, input_b}, output);
    for (float f : output->get_data<float>()) {
        std::cout << f << " ";
    }
    std::cout << std::endl;

}
*/

