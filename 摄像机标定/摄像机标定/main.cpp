////#include <opencv2/calib3d/calib3d.hpp>
////#include <opencv2/highgui/highgui.hpp>
////#include <iostream>
////using namespace std;
////using namespace cv;
//
////int main(){
////  VideoCapture cap(1);
////  cap.set(CV_CAP_PROP_FRAME_WIDTH,640);
////  cap.set(CV_CAP_PROP_FRAME_HEIGHT,480);
////   if(!cap.isOpened()){
////	  cout<<"fail to open camera"<<endl;
////   }
////   namedWindow("Calibration");
////   cout<<"Press G to start capturing images!"<<endl;
////   char pressedkey;
////   cin>>pressedkey;
////   if (cap.isOpened()&&pressedkey=='g')
////   
////	   cap.
////   }
////}
//
//
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include  <Eigen/Dense>
using namespace cv;
using namespace std;
#pragma  region

//#pragma  region
//
//
//enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };
//int main()
//{ 
//    /************************************************************************  
//           从摄像机中读取多幅图像,从中提取出角点，然后对角点进行亚像素精确化 
//    *************************************************************************/ 
//	int image_count=  10;                    /****    图像数量     ****/  
//	Mat frame;
//	Size image_size;                         /****     图像的尺寸      ****/   
//	Size board_size = Size(9,6);            /****    定标板上每行、列的角点数       ****/  
//	vector<Point2f> corners;                  /****    缓存每幅图像上检测到的角点       ****/
//	vector<vector<Point2f>>  corners_Seq;    /****  保存检测到的所有角点       ****/   
//	ofstream fout("calibration_result.txt");  /**    保存定标结果的文件     **/
//	int mode = DETECTION;
//
//	VideoCapture cap(1);
//	cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
//	cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);
//	if(!cap.isOpened()){
//		std::cout<<"打开摄像头失败，退出";
//		exit(-1);
//	}
//	namedWindow("Calibration");
//    std::cout<<"Press 'g' to start capturing images!"<<endl;  
//
//	int count = 0,n=0;
//	stringstream tempname;
//	string filename;
//	int key;
//	string msg;
//	int baseLine;
//	Size textSize;
//	while(n < image_count )
//	{
//		frame.setTo(0);
//		cap>>frame;
//		if(mode == DETECTION)
//		{
//			key = 0xff & waitKey(30);
//			if( (key & 255) == 27 ){
//				
//				break;
//			}
//			if( cap.isOpened() && key == 'g' )
//			{
//				cout<<"ggg"<<endl;
//				mode = CAPTURING;
//			}
//		}
//
//		if(mode == CAPTURING)
//		{
//			cout<<"mode==CAPUTRING"<<endl;
//			key = 0xff & waitKey(30);
//			cout<<key<<endl;
//			cout<<(key&255)<<endl;
//			if( (key & 255) == 255 )
//			{
//				cout<<"if( (key & 255) == 32 )"<<endl;
//				image_size = frame.size();
//				/* 提取角点 */   
//				Mat imageGray;
//				cvtColor(frame, imageGray , CV_RGB2GRAY);
//				bool patternfound = findChessboardCorners(frame, board_size, corners,CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK );
//				if (patternfound)   
//				{    
//					n++;
//					tempname<<n;
//					tempname>>filename;
//					filename+=".jpg";
//					/* 亚像素精确化 */
//					cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
//					count += corners.size();
//					corners_Seq.push_back(corners);
//					imwrite(filename,frame);
//					tempname.clear();
//					filename.clear();
//				}
//				else
//				{
//					std::cout<<"Detect Failed.\n";
//				}
//			}			
//		}
//		msg = mode == CAPTURING ? "100/100/s" : mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
//		baseLine = 0;
//		textSize = getTextSize(msg, 1, 1, 1, &baseLine);
//		Point textOrigin(frame.cols - 2*textSize.width - 10, frame.rows - 2*baseLine - 10);
//
//		if( mode == CAPTURING )
//		{
//			msg = format( "%d/%d",n,image_count);
//		}
//
//		putText( frame, msg, textOrigin, 1, 1,mode != CALIBRATED ? Scalar(0,0,255) : Scalar(0,255,0));
//
//		imshow("Calibration",frame);
//		key = 0xff & waitKey(1);
//		if( (key & 255) == 27 )
//			break;
//    }   
//
//    std::cout<<"角点提取完成！\n"; 
//
//    /************************************************************************  
//           摄像机定标  
//    *************************************************************************/   
//    std::cout<<"开始定标………………"<<endl;   
//    Size square_size = Size(25,25);                                      /**** 实际测量得到的定标板上每个棋盘格的大小   ****/  
//	vector<vector<Point3f>>  object_Points;                                      /****  保存定标板上角点的三维坐标   ****/
//
//    Mat image_points = Mat(1, count , CV_32FC2, Scalar::all(0));          /*****   保存提取的所有角点   *****/   
//	vector<int>  point_counts;                                          /*****    每幅图像中角点的数量    ****/   
//	Mat intrinsic_matrix = Mat(3,3, CV_32FC1, Scalar::all(0));                /*****    摄像机内参数矩阵    ****/   
//    Mat distortion_coeffs = Mat(1,5, CV_32FC1, Scalar::all(0));            /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */ 
//    vector<Mat> rotation_vectors;                                      /* 每幅图像的旋转向量 */  
//	vector<Mat> translation_vectors;                                  /* 每幅图像的平移向量 */  
//     
//    /* 初始化定标板上角点的三维坐标 */     
//    for (int t=0;t<image_count;t++) 
//	{   
//		vector<Point3f> tempPointSet;
//        for (int i=0;i<board_size.height;i++) 
//		{   
//            for (int j=0;j<board_size.width;j++) 
//			{   
//                /* 假设定标板放在世界坐标系中z=0的平面上 */   
//				Point3f tempPoint;
//				tempPoint.x = i*square_size.width;
//				tempPoint.y = j*square_size.height;
//				tempPoint.z = 0;
//				tempPointSet.push_back(tempPoint);
//            }   
//        }
//		object_Points.push_back(tempPointSet);
//    }   
//   
//    /* 初始化每幅图像中的角点数，这里我们假设每幅图像中都可以看到完整的定标板 */   
//    for (int i=0; i< image_count; i++)   
//	{
//        point_counts.push_back(board_size.width*board_size.height);   
//	}
//       
//    /* 开始定标 */   
//    calibrateCamera(object_Points, corners_Seq, image_size,  intrinsic_matrix  ,distortion_coeffs, rotation_vectors, translation_vectors);   
//    std::cout<<"定标完成！\n";   
//       
//    /************************************************************************  
//           对定标结果进行评价  
//    *************************************************************************/   
//    std::cout<<"开始评价定标结果………………"<<endl;   
//    double total_err = 0.0;                   /* 所有图像的平均误差的总和 */   
//    double err = 0.0;                        /* 每幅图像的平均误差 */   
//	vector<Point2f>  image_points2;             /****   保存重新计算得到的投影点    ****/   
//   
//    std::cout<<"每幅图像的定标误差："<<endl;   
//    fout<<"每幅图像的定标误差："<<endl<<endl;   
//    for (int i=0;  i<image_count;  i++) 
//	{
//		vector<Point3f> tempPointSet = object_Points[i];
//        /****    通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点     ****/
//		projectPoints(tempPointSet, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs, image_points2);
//        /* 计算新的投影点和旧的投影点之间的误差*/  
//		vector<Point2f> tempImagePoint = corners_Seq[i];
//		Mat tempImagePointMat = Mat(1,tempImagePoint.size(),CV_32FC2);
//		Mat image_points2Mat = Mat(1,image_points2.size(), CV_32FC2);
//		for (int j = 0 ; j < tempImagePoint.size(); j++)
//		{
//			image_points2Mat.at<Vec2f>(0,j) = Vec2f(image_points2[j].x, image_points2[j].y);
//			tempImagePointMat.at<Vec2f>(0,j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
//		}
//		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
//        total_err += err/=  point_counts[i];   
//        std::cout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;   
//        fout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;   
//    }   
//    std::cout<<"总体平均误差："<<total_err/image_count<<"像素"<<endl;   
//    fout<<"总体平均误差："<<total_err/image_count<<"像素"<<endl<<endl;   
//    std::cout<<"评价完成！"<<endl;   
//   
//    /************************************************************************  
//           保存定标结果  
//    *************************************************************************/   
//    std::cout<<"开始保存定标结果………………"<<endl;       
//    Mat rotation_matrix = Mat(3,3,CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */   
//       
//    fout<<"相机内参数矩阵："<<endl;   
//    fout<<intrinsic_matrix<<endl<<endl;   
//    fout<<"畸变系数：\n";   
//    fout<<distortion_coeffs<<endl<<endl<<endl;   
//    for (int i=0; i<image_count; i++) 
//	{ 
//        fout<<"第"<<i+1<<"幅图像的旋转向量："<<endl;   
//        fout<<rotation_vectors[i]<<endl;   
//  
//        /* 将旋转向量转换为相对应的旋转矩阵 */   
//        Rodrigues(rotation_vectors[i],rotation_matrix);   
//        fout<<"第"<<i+1<<"幅图像的旋转矩阵："<<endl;   
//        fout<<rotation_matrix<<endl;   
//        fout<<"第"<<i+1<<"幅图像的平移向量："<<endl;   
//        fout<<translation_vectors[i]<<endl<<endl;   
//    }   
//    std::cout<<"完成保存"<<endl; 
//	fout<<endl;
//
//	/************************************************************************  
//           显示定标结果  
//    *************************************************************************/
// 	Mat mapx = Mat(image_size,CV_32FC1);
// 	Mat mapy = Mat(image_size,CV_32FC1);
// 	Mat R = Mat::eye(3,3,CV_32F);
// 	std::cout<<"保存矫正图像"<<endl;
// 	string imageFileName;
// 	std::stringstream StrStm;
// 	for (int i = 0 ; i != image_count ; i++)
// 	{
// 		std::cout<<"Frame #"<<i+1<<"..."<<endl;
// 		Mat newCameraMatrix = Mat(3,3,CV_32FC1,Scalar::all(0));
// 		initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,intrinsic_matrix,image_size,CV_32FC1,mapx,mapy);
// 		StrStm.clear();
// 		imageFileName.clear();
// 		StrStm<<i+1;
// 		StrStm>>imageFileName;
// 		imageFileName += ".jpg";
// 		Mat t = imread(imageFileName);
// 		Mat newimage = t.clone();
// 		cv::remap(t,newimage,mapx, mapy, INTER_LINEAR);
// 		StrStm.clear();
// 		imageFileName.clear();
// 		StrStm<<i+1;
// 		StrStm>>imageFileName;
// 		imageFileName += "_d.jpg";
// 		imwrite(imageFileName,newimage);
// 	}
// 	std::cout<<"保存结束"<<endl;
//	return 0;
//}
//
//#pragma  endregion

#pragma   endregion
//#include <iostream>
//#include<string>
//#include <sstream>
//using namespace std;
//int main()
//{
//	for (int i =0;i<10;i++)
//	{
//		string aa="第";
//		stringstream ss;
//		string number;
//		ss<<i;
//		
//		ss>>number;
//		aa+=number;
//		aa+="张";
//			cout<<aa<<endl;
//	}
//	
//}
#include <sstream>
#include <string>
using Eigen::MatrixXd;
int main()

{
	//int key=0;
	//waitKey(0);//只对namedwindow函数创造出来的窗口有效
	//for(int i =0;i<5;i++){
	//	key=waitKey(10000000000);
	//}
	//VideoCapture cap(0)
    
	//cout<<key<<endl;
	
	MatrixXd m(2,2);
	m(0,0) = 3;
	m(1,0) = 2.5;
	m(0,1) = -1;
	m(1,1) = m(1,0) + m(0,1);
	std::cout << m << std::endl;


	string pic_num;
	string  dirpre="";
	vector<Point2f> img_corner;
	vector<vector<Point2f>> img_corners;
	vector<Point3f> world_corner;
	vector<vector<Point3f>> world_corners;
	Size img_size;
	for(int i=0;i<10;i++){
		stringstream trans;
		trans<<i;
		trans>>pic_num;
		//dir.append(pic_num).append(".jpg");
		string dir="";

		dirpre="C:\\Users\\Administrator\\Downloads\\amcap\\test\\";
		dir=dirpre.append(pic_num).append(".jpg");
		cout<<pic_num<<endl;
		cout<<dir<<endl;
	    Mat src=imread(dir,1);
		img_size=src.size();
		Mat src_g=imread(dir,0);
		//cout<<dir.append(pic_num)<<endl;
//		imshow("1",src);

		/*	bool iffind=;
		cout<<iffind<<endl;*/
		//find the img  corners
		if (findChessboardCorners(src,Size(7,7),img_corner,CV_CALIB_CB_ADAPTIVE_THRESH))
		{
			//cout<<"success"<<endl;
			//draw the corners
			//cout<<corners.at(i)<<endl;
			
			cornerSubPix(src_g,img_corner,Size(5,5),Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			//cout<<corners.at(i)<<endl;
			
			
			// draw corners
	/*		for (int i=0;i<img_corner.size();i++)
			{
				circle(src,img_corner.at(i),5,Scalar(255,0,255),2);
			}
			*/
			/*	drawChessboardCorners(src,Size(7,7),img_corner,1);
			imshow(dir,src_g);
			imshow("原图",src);
			waitKey(0);*/

			img_corners.push_back(img_corner);
			
		}

		else
		{

			cout<<"failed"<<pic_num<<endl;
		}

	}
	//find the world corners
	//先x再y
	
	for (int i=0;i<10;i++)
	{
		world_corner.clear();
		for (int j =0;j<7;j++)
		{
			for (int k=0;k<7;k++)
			{
				Point3f  world_point(j,k,0);
				world_corner.push_back(world_point);
			}
		}
		world_corners.push_back(world_corner);
		/*cout<<format(world_corners[i],"numpy")<<endl;
		cout<<format(img_corners[i],"numpy")<<endl;*/
	
	}
	/*cout<<"world_corners"<<world_corners.size()<<endl;
	cout<<"world_corner"<<world_corner.size()<<endl;
	cout<<"img_corners"<<img_corners.size()<<endl;
	cout<<"img_corner"<<img_corner.size()<<endl;*/
	cout<<"calibration start"<<endl;
	//开始标定
	//Mat intrinsic_matrix;
	//Mat  distortion_coeffs;
	// Mat intrinsic_matrix = Mat(3,3, CV_32FC1, Scalar::all(0));                /*****    摄像机内参数矩阵    ****/   
 //   Mat distortion_coeffs = Mat(1,5, CV_32FC1, Scalar::all(0)); 
	Mat intrinsic_matrix;
	Mat distortion_coeffs;
	vector<Mat> translation_vector;  //wrong :  Mat translation_vector;
	vector<Mat> rotation_vector;
	//cout<<"world_corners"<<world_corners.size()<<endl;
	//cout<<"img_corners"<<img_corners.size()<<endl;
	//cout<<"world_corner"<<world_corner.size()<<endl;
	//cout<<"img_corners"<<img_corner.size()<<endl;
	//cout<<img_size.height<<"   "<<img_size.width<<endl;
	//calibrateCamera(world_corners,img_corners,img_size,intrinsic_matrix,distortion_coeffs,rotation_vector,translation_vetor);
	
	//double e=calibrateCamera(world_corners, img_corners, img_size,  intrinsic_matrix  ,distortion_coeffs, rotation_vector, translation_vector,14|2048|4096);   
	double e=calibrateCamera(world_corners, img_corners, img_size,  intrinsic_matrix  ,distortion_coeffs, rotation_vector, translation_vector);   
	cout<<"biaodingwancheng"<<e<<endl;
	/*for (int i=0;i<10;i++)
	{
	cout<<"world_corners"<<format(world_corners[i],"numpy")<<endl;
	}*/

	//Mat aa=Mat(3,3,CV_32FC1);
	//aa.at<float>(0,0)=1.0f;
	//aa.at<float>(0,1)=2.0f;
	//aa.at<float>(0,2)=3.0f;

	//aa.at<float>(1,0)=4.0f;
	//aa.at<float>(1,1)=5.0f;
	//aa.at<float>(1,2)=6.0f;
	//aa.at<float>(2,0)=7.0f;
	//aa.at<float>(2,1)=8.0f;
	//aa.at<float>(2,2)=9.0f;
	//Mat bb=Mat(3,3,CV_32FC1);
	//normalize(aa,aa,1,0,NORM_MINMAX);
 // //	cout<<"sss %d" <<11<<endl;
	//double cc=norm(aa,bb,NORM_L2);
	////cout<<format(aa,"python")<<endl;
	////cout<<format(cc,"numpy")<<endl;
	//cout<<cc<<endl;
	//Mat tempImagePoint=aa;
	//cout<<rotation_vector.size()<<endl;
	//Mat tempImagePointMat = Mat(1,rotation_vector.size(),CV_32FC2);
	//for(int i=0;i<aa.size().width;i++)
	//{

	//	for (int j=0;j<aa.size().height;j++)
	//	{
	//		cout<<aa.at<float>(i,j)<<endl;
	//	}
	//}
	vector<Point2f> img_points2;
	Mat  pImg_points;
	Mat  oImg_points;
	double err=0.0;
	double totalerr=0.0;
	for(int i=0;i<10;i++)
	{

		projectPoints(world_corners[i],rotation_vector[i],translation_vector[i],intrinsic_matrix,distortion_coeffs,img_points2);
		pImg_points=Mat(1,img_points2.size(),CV_32FC2);
		oImg_points=Mat(1,world_corners[i].size(),CV_32FC2);
		for(int j=0;j<img_points2.size();j++)
		{
			//cout<<"img_points2"<<format(img_points2,"numpy")<<endl;
			pImg_points.at<Vec2f>(0,j)=Vec2f(img_points2[j].x,img_points2[j].y);
			oImg_points.at<Vec2f>(0,j)=Vec2f(world_corners[i][j].x,world_corners[i][j].y);
		}
	//	cout<<"pImg_points"<<format(pImg_points,"numpy")<<endl;
	//	cout<<"oImg_points"<<format(oImg_points,"numpy")<<endl;
		err=norm(pImg_points,oImg_points,NORM_L2);
		 totalerr += err/=  49;   
		cout<<"di"<<i<<"幅图片的误差为"<<err<<endl;

	}
	cout<<img_points2.size()<<endl;
	cout<<"总体误差为"<<totalerr/10<<endl;

	ofstream  fout("C:\\Users\\Administrator\\Downloads\\amcap\\test\\intrinsic_matrix.txt");

	fout<<format(intrinsic_matrix,"numpy")<<endl;
	fout<<"aa1"<<endl;

	

	return 0;
}
vector<vector<Point2f>> getImgCorners()
{
	string pic_num;
	string  dirpre="";
	vector<Point2f> img_corner;
	vector<vector<Point2f>> img_corners;
		Size img_size;
	for(int i=0;i<10;i++){
		stringstream trans;
		trans<<i;
		trans>>pic_num;
		//dir.append(pic_num).append(".jpg");
		string dir="";

		dirpre="C:\\Users\\Administrator\\Downloads\\amcap\\test\\";
		dir=dirpre.append(pic_num).append(".jpg");
		cout<<pic_num<<endl;
		cout<<dir<<endl;
	    Mat src=imread(dir,1);
		img_size=src.size();
		Mat src_g=imread(dir,0);
		//cout<<dir.append(pic_num)<<endl;
//		imshow("1",src);

		/*	bool iffind=;
		cout<<iffind<<endl;*/
		//find the img  corners
		if (findChessboardCorners(src,Size(7,7),img_corner,CV_CALIB_CB_ADAPTIVE_THRESH))
		{
			//cout<<"success"<<endl;
			//draw the corners
			//cout<<corners.at(i)<<endl;
			cornerSubPix(src_g,img_corner,Size(5,5),Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			//cout<<corners.at(i)<<endl;
			
			
			// draw corners
		/*	for (int i=0;i<corners.size();i++)
			{
				circle(src,corners.at(i),5,Scalar(255,0,255),2);
			}
			
			drawChessboardCorners(src,Size(7,7),corners,1);
			imshow(dir,src_g);
			imshow("原图",src);
			waitKey(0);*/

			img_corners.push_back(img_corner);
		}

		else
		{

			cout<<"failed"<<pic_num<<endl;
		}

	}
	return img_corners;

}

vector<vector<Point3f>> getWorldCorners()
{
	vector<Point3f> world_corner;
	vector<vector<Point3f>> world_corners;

	for (int i=0;i<10;i++)
	{
		world_corner.clear();
		for (int j =0;j<7;j++)
		{
			for (int k=0;k<7;k++)
			{
				Point3f  world_point(i,j,0);
				world_corner.push_back(world_point);
			}
		}
		world_corners.push_back(world_corner);
	}
	return world_corners;

}
Size getImgSize()
{
	Size img_size;
	Mat src= imread("C:\\Users\\Administrator\\Downloads\\amcap\\test\\1.jpg");
	return src.size();
}
int main1()
{
	string pic_num;
	string  dirpre="";
	vector<Point2f> img_corner;
	vector<vector<Point2f>> img_corners;
	vector<Point3f> world_corner;
	vector<vector<Point3f>> world_corners;
	Size img_size;
	for(int i=0;i<10;i++){
		stringstream trans;
		trans<<i;
		trans>>pic_num;
		//dir.append(pic_num).append(".jpg");
		string dir="";

		dirpre="C:\\Users\\Administrator\\Downloads\\amcap\\test\\";
		dir=dirpre.append(pic_num).append(".jpg");
		cout<<pic_num<<endl;
		cout<<dir<<endl;
	    Mat src=imread(dir,1);
		img_size=src.size();
		Mat src_g=imread(dir,0);
		//cout<<dir.append(pic_num)<<endl;
//		imshow("1",src);

		/*	bool iffind=;
		cout<<iffind<<endl;*/
		//find the img  corners
		if (findChessboardCorners(src,Size(7,7),img_corner,CV_CALIB_CB_ADAPTIVE_THRESH))
		{
			//cout<<"success"<<endl;
			//draw the corners
			//cout<<corners.at(i)<<endl;
			cornerSubPix(src_g,img_corner,Size(5,5),Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			//cout<<corners.at(i)<<endl;
			
			
			// draw corners
		/*	for (int i=0;i<corners.size();i++)
			{
				circle(src,corners.at(i),5,Scalar(255,0,255),2);
			}
			
			drawChessboardCorners(src,Size(7,7),corners,1);
			imshow(dir,src_g);
			imshow("原图",src);
			waitKey(0);*/

			img_corners.push_back(img_corner);
		}

		else
		{

			cout<<"failed"<<pic_num<<endl;
		}

	}
	/************************************************************************  
	//           从摄像机中读取多幅图像,从中提取出角点，然后对角点进行亚像素精确化 
	//    *************************************************************************/ 
	//	int image_count=  10;                    /****    图像数量     ****/  
	//	Mat frame;
	//	Size image_size;                         /****     图像的尺寸      ****/   
	//	Size board_size = Size(9,6);            /****    定标板上每行、列的角点数       ****/  
	//	vector<Point2f> corners;                  /****    缓存每幅图像上检测到的角点       ****/
	//	vector<vector<Point2f>>  corners_Seq;    /****  保存检测到的所有角点       ****/   
	//	ofstream fout("calibration_result.txt");  /**    保存定标结果的文件     **/
	//	int mode = DETECTION;
	//
	//	VideoCapture cap(1);
	//	cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
	//	cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);
	//	if(!cap.isOpened()){
	//		std::cout<<"打开摄像头失败，退出";
	//		exit(-1);
	//	}
	//	namedWindow("Calibration");
	//    std::cout<<"Press 'g' to start capturing images!"<<endl;  
	//
	//	int count = 0,n=0;
	//	stringstream tempname;
	//	string filename;
	//	int key;
	//	string msg;
	//	int baseLine;
	//	Size textSize;
	//	while(n < image_count )
	//	{
	//		frame.setTo(0);
	//		cap>>frame;
	//		if(mode == DETECTION)
	//		{
	//			key = 0xff & waitKey(30);
	//			if( (key & 255) == 27 ){
	//				
	//				break;
	//			}
	//			if( cap.isOpened() && key == 'g' )
	//			{
	//				cout<<"ggg"<<endl;
	//				mode = CAPTURING;
	//			}
	//		}
	//
	//		if(mode == CAPTURING)
	//		{
	//			cout<<"mode==CAPUTRING"<<endl;
	//			key = 0xff & waitKey(30);
	//			cout<<key<<endl;
	//			cout<<(key&255)<<endl;
	//			if( (key & 255) == 255 )
	//			{
	//				cout<<"if( (key & 255) == 32 )"<<endl;
	//				image_size = frame.size();
	//				/* 提取角点 */   
	//				Mat imageGray;
	//				cvtColor(frame, imageGray , CV_RGB2GRAY);
	//				bool patternfound = findChessboardCorners(frame, board_size, corners,CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK );
	//				if (patternfound)   
	//				{    
	//					n++;
	//					tempname<<n;
	//					tempname>>filename;
	//					filename+=".jpg";
	//					/* 亚像素精确化 */
	//					cornerSubPix(imageGray, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	//					count += corners.size();
	//					corners_Seq.push_back(corners);
	//					imwrite(filename,frame);
	//					tempname.clear();
	//					filename.clear();
	//				}
	//				else
	//				{
	//					std::cout<<"Detect Failed.\n";
	//				}
	//			}			
	//		}
	//		msg = mode == CAPTURING ? "100/100/s" : mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
	//		baseLine = 0;
	//		textSize = getTextSize(msg, 1, 1, 1, &baseLine);
	//		Point textOrigin(frame.cols - 2*textSize.width - 10, frame.rows - 2*baseLine - 10);
	//
	//		if( mode == CAPTURING )
	//		{
	//			msg = format( "%d/%d",n,image_count);
	//		}
	//
	//		putText( frame, msg, textOrigin, 1, 1,mode != CALIBRATED ? Scalar(0,0,255) : Scalar(0,255,0));
	//
	//		imshow("Calibration",frame);
	//		key = 0xff & waitKey(1);
	//		if( (key & 255) == 27 )
	//			break;
	//    }   
	//
	//    std::cout<<"角点提取完成！\n"; 
	//
	//    /************************************************************************  
	//           摄像机定标  
	//    *************************************************************************/   
	    std::cout<<"开始定标………………"<<endl;   
	  //  Size square_size = Size(25,25);                                      /**** 实际测量得到的定标板上每个棋盘格的大小   ****/  
//		vector<vector<Point3f>>  object_Points;                                      /****  保存定标板上角点的三维坐标   ****/
	//    Size board_size(7,7);
	 //   Mat image_points = Mat(1, count , CV_32FC2, Scalar::all(0));          /*****   保存提取的所有角点   *****/   
	//	vector<int>  point_counts;                                          /*****    每幅图像中角点的数量    ****/   
		//Mat intrinsic_matrix = Mat(3,3, CV_32FC1, Scalar::all(0));                /*****    摄像机内参数矩阵    ****/   
	    Mat intrinsic_matrix;
	//	Mat distortion_coeffs = Mat(1,5, CV_32FC1, Scalar::all(0));            /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */ 
	    Mat distortion_coeffs;
		vector<Mat> rotation_vectors;                                      /* 每幅图像的旋转向量 */  
		vector<Mat> translation_vectors;                                  /* 每幅图像的平移向量 */  
	 //    int image_count=10;
	    /* 初始化定标板上角点的三维坐标 */ 
		
		 for (int i=0;i<10;i++)
		 {
			 world_corner.clear();
			 for (int j =0;j<7;j++)
			 {
				 for (int k=0;k<7;k++)
				 {
					 Point3f  world_point(i,j,0);
					 world_corner.push_back(world_point);
				 }
			 }
			 world_corners.push_back(world_corner);
		 }
	 //   for (int t=0;t<image_count;t++) 
		//{   
		//	vector<Point3f> tempPointSet;
	 //       for (int i=0;i<board_size.height;i++) 
		//{   
	 //           for (int j=0;j<board_size.width;j++) 
		//		{   
	 //               /* 假设定标板放在世界坐标系中z=0的平面上 */   
		//			Point3f tempPoint;
		//			tempPoint.x = i*square_size.width;
		//			tempPoint.y = j*square_size.height;
		//			tempPoint.z = 0;
		//			tempPointSet.push_back(tempPoint);
		//		}   
		//	}
		//	object_Points.push_back(tempPointSet);
		//}   
	//   
	//    /* 初始化每幅图像中的角点数，这里我们假设每幅图像中都可以看到完整的定标板 */   
	//    for (int i=0; i< image_count; i++)   
	//	{
	//        point_counts.push_back(board_size.width*board_size.height);   
	//	}
	//       
	//    /* 开始定标 */   
	    calibrateCamera(world_corners, img_corners, img_size,  intrinsic_matrix  ,distortion_coeffs, rotation_vectors, translation_vectors);   
	    
		std::cout<<"定标完成！\n";   
	//       
	//    /************************************************************************  
	//           对定标结果进行评价  
	//    *************************************************************************/   
	//    std::cout<<"开始评价定标结果………………"<<endl;   
	//    double total_err = 0.0;                   /* 所有图像的平均误差的总和 */   
	//    double err = 0.0;                        /* 每幅图像的平均误差 */   
	//	vector<Point2f>  image_points2;             /****   保存重新计算得到的投影点    ****/   
	//   
	//    std::cout<<"每幅图像的定标误差："<<endl;   
	//    fout<<"每幅图像的定标误差："<<endl<<endl;   
	//    for (int i=0;  i<image_count;  i++) 
	//	{
	//		vector<Point3f> tempPointSet = object_Points[i];
	//        /****    通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点     ****/
	//		projectPoints(tempPointSet, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs, image_points2);
	//        /* 计算新的投影点和旧的投影点之间的误差*/  
	//		vector<Point2f> tempImagePoint = corners_Seq[i];
	//		Mat tempImagePointMat = Mat(1,tempImagePoint.size(),CV_32FC2);
	//		Mat image_points2Mat = Mat(1,image_points2.size(), CV_32FC2);
	//		for (int j = 0 ; j < tempImagePoint.size(); j++)
	//		{
	//			image_points2Mat.at<Vec2f>(0,j) = Vec2f(image_points2[j].x, image_points2[j].y);
	//			tempImagePointMat.at<Vec2f>(0,j) = Vec2f(tempImagePoint[j].x, tempImagePoint[j].y);
	//		}
	//		err = norm(image_points2Mat, tempImagePointMat, NORM_L2);
	//        total_err += err/=  point_counts[i];   
	//        std::cout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;   
	//        fout<<"第"<<i+1<<"幅图像的平均误差："<<err<<"像素"<<endl;   
	//    }   
	//    std::cout<<"总体平均误差："<<total_err/image_count<<"像素"<<endl;   
	//    fout<<"总体平均误差："<<total_err/image_count<<"像素"<<endl<<endl;   
	//    std::cout<<"评价完成！"<<endl;   
	//   
	//    /************************************************************************  
	//           保存定标结果  
	//    *************************************************************************/   
	//    std::cout<<"开始保存定标结果………………"<<endl;       
	//    Mat rotation_matrix = Mat(3,3,CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */   
	//       
	//    fout<<"相机内参数矩阵："<<endl;   
	//    fout<<intrinsic_matrix<<endl<<endl;   
	//    fout<<"畸变系数：\n";   
	//    fout<<distortion_coeffs<<endl<<endl<<endl;   
	//    for (int i=0; i<image_count; i++) 
	//	{ 
	//        fout<<"第"<<i+1<<"幅图像的旋转向量："<<endl;   
	//        fout<<rotation_vectors[i]<<endl;   
	//  
	//        /* 将旋转向量转换为相对应的旋转矩阵 */   
	//        Rodrigues(rotation_vectors[i],rotation_matrix);   
	//        fout<<"第"<<i+1<<"幅图像的旋转矩阵："<<endl;   
	//        fout<<rotation_matrix<<endl;   
	//        fout<<"第"<<i+1<<"幅图像的平移向量："<<endl;   
	//        fout<<translation_vectors[i]<<endl<<endl;   
	//    }   
	//    std::cout<<"完成保存"<<endl; 
	//	fout<<endl;
	//
	//	/************************************************************************  
	//           显示定标结果  
	//    *************************************************************************/
	// 	Mat mapx = Mat(image_size,CV_32FC1);
	// 	Mat mapy = Mat(image_size,CV_32FC1);
	// 	Mat R = Mat::eye(3,3,CV_32F);
	// 	std::cout<<"保存矫正图像"<<endl;
	// 	string imageFileName;
	// 	std::stringstream StrStm;
	// 	for (int i = 0 ; i != image_count ; i++)
	// 	{
	// 		std::cout<<"Frame #"<<i+1<<"..."<<endl;
	// 		Mat newCameraMatrix = Mat(3,3,CV_32FC1,Scalar::all(0));
	// 		initUndistortRectifyMap(intrinsic_matrix,distortion_coeffs,R,intrinsic_matrix,image_size,CV_32FC1,mapx,mapy);
	// 		StrStm.clear();
	// 		imageFileName.clear();
	// 		StrStm<<i+1;
	// 		StrStm>>imageFileName;
	// 		imageFileName += ".jpg";
	// 		Mat t = imread(imageFileName);
	// 		Mat newimage = t.clone();
	// 		cv::remap(t,newimage,mapx, mapy, INTER_LINEAR);
	// 		StrStm.clear();
	// 		imageFileName.clear();
	// 		StrStm<<i+1;
	// 		StrStm>>imageFileName;
	// 		imageFileName += "_d.jpg";
	// 		imwrite(imageFileName,newimage);
	// 	}
	// 	std::cout<<"保存结束"<<endl;
		return 0;

}

int main2()
{
	string pic_num;
	string  dirpre="";
	vector<Point2f> img_corner;
	vector<vector<Point2f>> img_corners;
	vector<Point3f> world_corner;
	vector<vector<Point3f>> world_corners;
	Size img_size;

	for(int i=0;i<10;i++){
		stringstream trans;
		trans<<i;
		trans>>pic_num;
		//dir.append(pic_num).append(".jpg");
		string dir="";

		dirpre="C:\\Users\\Administrator\\Downloads\\amcap\\test\\";
		dir=dirpre.append(pic_num).append(".jpg");
		cout<<pic_num<<endl;
		cout<<dir<<endl;
	    Mat src=imread(dir,1);
		img_size=src.size();
		Mat src_g=imread(dir,0);
		//cout<<dir.append(pic_num)<<endl;
//		imshow("1",src);

		/*	bool iffind=;
		cout<<iffind<<endl;*/
		//find the img  corners
		if (findChessboardCorners(src,Size(7,7),img_corner,CV_CALIB_CB_ADAPTIVE_THRESH))
		{
			//cout<<"success"<<endl;
			//draw the corners
			//cout<<corners.at(i)<<endl;
			cornerSubPix(src_g,img_corner,Size(5,5),Size(-1,-1),TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			//cout<<corners.at(i)<<endl;
			
			
			// draw corners
		/*	for (int i=0;i<corners.size();i++)
			{
				circle(src,corners.at(i),5,Scalar(255,0,255),2);
			}
			
			drawChessboardCorners(src,Size(7,7),corners,1);
			imshow(dir,src_g);
			imshow("原图",src);
			waitKey(0);*/

			img_corners.push_back(img_corner);
		}

		else
		{

			cout<<"failed"<<pic_num<<endl;
		}

	}

	for (int i=0;i<10;i++)
	{
		world_corner.clear();
		for (int j =0;j<7;j++)
		{
			for (int k=0;k<7;k++)
			{
				Point3f  world_point(i,j,0);
				world_corner.push_back(world_point);
			}
		}
		world_corners.push_back(world_corner);
	}
	Mat intrinsic_matrix = Mat(3,3, CV_32FC1, Scalar::all(0));                /*****    摄像机内参数矩阵    ****/   
	Mat distortion_coeffs = Mat(1,5, CV_32FC1, Scalar::all(0)); 
	Mat translation_vector;
	Mat rotation_vector;
	//cout<<"world_corners"<<world_corners.size()<<endl;
	//cout<<"img_corners"<<img_corners.size()<<endl;
	//cout<<"world_corner"<<world_corner.size()<<endl;
	//cout<<"img_corners"<<img_corner.size()<<endl;
	//cout<<img_size.height<<"   "<<img_size.width<<endl;
	//calibrateCamera(world_corners,img_corners,img_size,intrinsic_matrix,distortion_coeffs,rotation_vector,translation_vetor);
	calibrateCamera(world_corners, img_corners, img_size,  intrinsic_matrix  ,distortion_coeffs, rotation_vector, translation_vector);   
	return 0;
}

int main33()
{
	vector<vector<Point2f>> ImgCorners=getImgCorners();
	vector<vector<Point3f>> WorldCorners=getWorldCorners();
	Mat intrinsic_matrix;
	//	Mat distortion_coeffs = Mat(1,5, CV_32FC1, Scalar::all(0));            /* 摄像机的5个畸变系数：k1,k2,p1,p2,k3 */ 
	Mat distortion_coeffs;
	vector<Mat> rotation_vectors;                                      /* 每幅图像的旋转向量 */  
	vector<Mat> translation_vectors;   
	Size imgSize=getImgSize();
	//calibrateCamera(WorldCorners,ImgCorners,imgSize,intrinsic_matrix,distortion_coeffs,translation_vectors,translation_vectors);
	calibrateCamera(WorldCorners, ImgCorners, imgSize,  intrinsic_matrix  ,distortion_coeffs, rotation_vectors, translation_vectors);   

	cout<<"OK"<<endl;

	return 0;

}