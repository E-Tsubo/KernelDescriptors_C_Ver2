//#define DEBUG
// probably have to use Eigen
#include "libkerneldesc.h"
//#include <unistd.h>
#include <boost/thread.hpp>
#include <boost/thread/locks.hpp>
#include <stdlib.h>

// for getting directory information
// needs a link to -lboost_filesystem and -lboost_system
#include <boost/filesystem/operations.hpp> // includes boost/filesystem/path.hpp
#include <boost/filesystem/fstream.hpp>

//using namespace  boost::filesystem;

static IplImage *frame = NULL;
static IplImage *frame_dep = NULL;

static boost::mutex m;
const char* model_name = model_names[MODEL_TYPE];
const char* model_file = model_files[MODEL_TYPE];
const char* model_var = model_vars[MODEL_TYPE];
const char* param_file = param_files[MODEL_TYPE];

#define USE_KINECT 1
#ifdef USE_KINECT
  #define USE_KINECT_RGB 1
  #define USE_KINECT_DEP 1
#endif

//either USE_SELFCUT or USE_GRABCUT
//Not supported both option at same time
#define USE_SELFCUT 1
#define	USE_GRABCUT 1
#ifdef USE_GRABCUT
  #include <opencv2/imgproc/imgproc.hpp>
#endif

#define FRAME_WIDTH_LIVE 160*4
#define FRAME_HEIGHT_LIVE 120*4

bool run_debug_mode=false;
bool run_self_cut=false;
bool run_grab_cut=false;
string object_name;
CvFont font;

CvCapture *capture = NULL;
#if USE_KINECT
  cv::VideoCapture capturekinect( CV_CAP_OPENNI );
#endif

void ClearCinBufferFlags()
{
  cin.clear();//opt = -1;
  cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

void SetupFont (CvFont& font)
{
  double hScale=0.8;
  double vScale=0.8;
  int    lineWidth=2;
  cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);
}

#if USE_KINECT
void kinectCapture()
{
  cv::Mat kinimg, kinimg_dep;
  IplImage tmp, tmp_dep;
  
  capturekinect.grab();
  capturekinect.retrieve( kinimg, CV_CAP_OPENNI_BGR_IMAGE );//CV_8UC3
  capturekinect.retrieve( kinimg_dep, CV_CAP_OPENNI_DEPTH_MAP );//CV_16UC1
  
  tmp = kinimg; tmp_dep = kinimg_dep;
  //if( FRAME_WIDTH_LIVE == 640 && FRAME_WIDTH_LIVE == 480 ){
    IplImage* out = cvCreateImage( cvSize( tmp.width, tmp.height ), IPL_DEPTH_8U, 3 );
    IplImage* out_dep = cvCreateImage( cvSize( tmp.width, tmp.height ), IPL_DEPTH_16U, 1 );
    cvCopy( &tmp, out, NULL );
    cvCopy( &tmp_dep, out_dep, NULL );
    frame = out; frame_dep = out_dep;
    //}else{
    //IplImage* out = cvCreateImage( cvSize( tmp.width/2, tmp.height/2 ), IPL_DEPTH_8U, 3 );
    //IplImage* out_dep = cvCreateImage( cvSize( tmp.width/2, tmp.height/2 ), IPL_DEPTH_16U, 1 );
    //cvResize( &tmp, out, CV_INTER_CUBIC );
    //cvResize( &tmp_dep, out_dep, CV_INTER_CUBIC );
    //frame = out; frame_dep = out_dep;
    //}
}
#endif

void ThreadCaptureFrame()
{
  cout << "Cam capture started..." << endl;
  
  int key = 0;
  //capture = cvCaptureFromCAM(0);
  
  cvNamedWindow("Camera View", CV_WINDOW_AUTOSIZE);
  while ((char)key != 'q')
    {
      {
	boost::mutex::scoped_lock lock(m);
	#if USE_KINECT
	kinectCapture();
	#else
	frame = cvQueryFrame(capture);
	#endif
      }
      if (!frame) break;
      cvPutText(frame, object_name.c_str(), cvPoint(10,20), &font,cvScalar(0,256,0));
      cvShowImage("Camera View", frame);
      key = cvWaitKey(1);
      //break;
      
    }
  cvReleaseImage(&frame);
  cvReleaseCapture(&capture);
  cout << "Cam capture ended" << endl;
  
}

void ThreadRunDescriptors()
{
  // rescale window size
  const double frame_ratio=FRAME_WIDTH_LIVE/160;
  
  // font stuff
  CvFont font;
  SetupFont(font);
  //

  cout << "Descriptor thread started..." << endl;
  int key = 0;
  KernelDescManager* kdm = new KernelDescManager(string(model_name), string(model_file), string(model_var),
						 string(param_file), MODEL_TYPE, MAX_IMAGE_SIZE);
  IplImage* img_src = NULL;
  IplImage* img_crop = NULL;
  IplImage* dep_src = NULL;//For kinect
  IplImage* dep_crop = NULL;//For kinect
  
  cvNamedWindow ("Cropped Frame", CV_WINDOW_AUTOSIZE);
  cvNamedWindow("Processed Frame", CV_WINDOW_AUTOSIZE);
  if( USE_KINECT_DEP )
    cvNamedWindow("Depth Frame", CV_WINDOW_AUTOSIZE);

  //For pcloudkdes
  VectorXf top_left(2);
  
  while ((char)key != 'q')
    {
      //cout<< "Looping" << endl;
      MatrixXf imfea2;
      
      {
	boost::mutex::scoped_lock lock(m);
	if (frame != NULL) { 
	  cvReleaseImage(&img_src);
	  img_src = cvCloneImage(frame);
	  if( USE_KINECT_DEP ){
	    cvReleaseImage(&dep_src);
	    dep_src = cvCloneImage(frame_dep);
	  }
	}
      }
      
      if (img_src != NULL)
	{
	  if (run_self_cut) {
	    //RGB
	    //const int cut_width = 100;
	    //const int cut_height = 150;
	    //DEPTH 320x280
	    //const int cut_width = 50;
	    //const int cut_height = 80;
	    //DEPTH 640x480
	    const int cut_width = 70;
	    const int cut_height = 120;
	    
	    cv::Mat image( img_src );
	    int xcenter = (int)(img_src->width/2);
	    int ycenter = (int)(img_src->height/2);
	    
	    cv::Rect bounding_rect( xcenter-cut_width/2, ycenter-cut_height/2,
				    cut_width, cut_height );
	    cvSetImageROI( img_src, bounding_rect );
	    img_crop=cvCreateImage( cvSize(bounding_rect.width,bounding_rect.height), IPL_DEPTH_8U, 3 );
	    cvCopy( img_src, img_crop, NULL );
	    cvResetImageROI( img_src );
	    if (USE_KINECT_DEP) {
	      cvSetImageROI( dep_src, bounding_rect );
	      dep_crop=cvCreateImage( cvSize(bounding_rect.width,bounding_rect.height), IPL_DEPTH_16U, 1 );
	      cvCopy( dep_src, dep_crop, NULL );
	      cvResetImageROI( dep_src );

	      top_left[0] = bounding_rect.x; top_left[1] = bounding_rect.y;
	    }
	    
	    //For Image View, draw rectangle which shows segmentation range.
	    cvRectangle( img_src, cvPoint( xcenter-cut_width/2, ycenter-cut_height/2 ),
			 cvPoint( xcenter+cut_width/2, ycenter+cut_height/2 ),
			 cvScalar( 0, 0, 255 ), 2, 8, 0 );
	    
	  }else if (run_grab_cut) {
	    // call opencv gracut to try and extract foreground object
	    cv::Mat image( img_src );
	    cv::Mat mask( img_src->height, img_src->width, CV_8UC1 );
	    cv::Mat bgdModel, fgdModel;
	    //cv::Mat bgdModel( img_src->height, img_src->width, CV_64FC1 );
	    //cv::Mat fgdModel( img_src->height, img_src->width, CV_64FC1 );
	    cv::Rect rect;
	    
	    const int margin_grabcut=max(5,(int)(5*frame_ratio));
	    const int centersize=max(5,(int)(5*frame_ratio));
	    
	    int height=img_src->height;
	    int width=img_src->width;
	    int ycenter=(int)(height/2);
	    int xcenter=(int)(width/2);
	    int startx=max( margin_grabcut, xcenter-min(ycenter,xcenter) );
	    int starty=max( margin_grabcut, ycenter-min(ycenter,xcenter) );
	    int endx=min( width-margin_grabcut, xcenter+min(ycenter,xcenter) );
	    int endy=min( height-margin_grabcut, ycenter+min(ycenter,xcenter) );
	    rect.x=startx; rect.y=starty;
	    rect.width=(endx-startx+1); rect.height=(endy-starty+1);
	    cv::grabCut( image, mask, rect, bgdModel, fgdModel, (1), cv::GC_INIT_WITH_RECT);
	    cv::Mat submask=mask( cv::Range(ycenter-centersize, ycenter+centersize),
				  cv::Range(xcenter-centersize,xcenter+centersize) );
	    submask=cv::GC_FGD;
	    //mask.at<unsigned char>( (int)(img_src->height/2), (int)(img_src->width/2) ) = cv::GC_FGD;
	    //cout << "center=" << (int)( mask.at<unsigned char>( ycenter,xcenter ) ) << endl;
	    cv::grabCut( image, mask, rect, bgdModel, fgdModel, (1), cv::GC_INIT_WITH_MASK);
	    mask = (mask & 1);
	    cv::Mat foreground(image.size(),CV_8UC3, cv::Scalar(255,255,255)); // all white image
	    image.copyTo(foreground,mask); // bg pixels not copied
	    // plot the in/outside boxes on the image
	    cvRectangle( &CvMat(foreground), cv::Point(xcenter-centersize,ycenter-centersize),
			 cv::Point(xcenter+centersize,ycenter+centersize), cv::Scalar(0,0,255) );
	    cvRectangle( &CvMat(foreground), cv::Point(startx,starty), cv::Point(endx,endy),
			 cv::Scalar(0,0,255) );
	    cvShowImage("GrabCut Masked", &IplImage(foreground) );
	    // find bounding rectangle
	    CvMemStorage* storage = cvCreateMemStorage();
	    CvSeq* contours;
	    cvFindContours( &CvMat(mask),storage,&contours);
	    
	    CvSeq* bestContour=NULL;
	    int maxlen=0;
	    for(; contours!=0; contours=contours->h_next) {
	      if (cvArcLength(contours)>maxlen) {
		maxlen=cvArcLength(contours);
		bestContour=contours;
	      }
	    }
	    cv::Rect bounding_rect = cvBoundingRect(bestContour);     // let's use the first contour
	    const int margin_rec=max(10,(int)(10*frame_ratio));
	    bounding_rect.x = max( bounding_rect.x-margin_rec, 1 );
	    bounding_rect.y -= margin_rec;
	    bounding_rect.width += 2*margin_rec;
	    bounding_rect.height += 2*margin_rec;
	    // check min. rectangle size
	    const int minrect=max(30,(int)(30*frame_ratio));
	    ycenter=bounding_rect.y+bounding_rect.height/2;
	    xcenter=bounding_rect.x+bounding_rect.width/2;
	    if (bounding_rect.width<minrect) {
	      int width_new = min( min( minrect, width-xcenter-1), xcenter-1 );
	      bounding_rect.x=bounding_rect.x-(width_new-bounding_rect.width)/2;
	      bounding_rect.width=width_new;
	    }
	    if (bounding_rect.height<minrect) {
	      int height_new = min( min( minrect, height-ycenter-1), ycenter-1 );
	      bounding_rect.y=bounding_rect.y-(height_new-bounding_rect.height)/2;
	      bounding_rect.height=height_new;
	    }
	    
	    // make rectangle fit
	    if ( bounding_rect.x<=0 ) {
	      bounding_rect.width += (bounding_rect.x-2);
	      bounding_rect.x =1;
	    }
	    if ( bounding_rect.y<=0 ) {
	      bounding_rect.height += (bounding_rect.y-2);
	      bounding_rect.y =1;
	    }
	    if ( bounding_rect.x+bounding_rect.width>=width-1 ) {
	      bounding_rect.width -= (bounding_rect.x+bounding_rect.width-width+2);
	    }
	    if ( bounding_rect.y+bounding_rect.height>=height-1 ) {
	      bounding_rect.height -= (bounding_rect.y+bounding_rect.height-height+2);
	    }
	    
	    //cout << "Rect=" << bounding_rect.width << " " << bounding_rect.height << " " << bounding_rect.x << " " << bounding_rect.y << " " << width << " " << height << endl;
	    cout << "Pre ROI:" << endl;
	    cvSetImageROI( img_src, bounding_rect );
	    cout << "Post ROI:" << endl;
	    img_crop=cvCreateImage( cvSize(bounding_rect.width,bounding_rect.height), IPL_DEPTH_8U, 3 );
	    cvCopy( img_src, img_crop, NULL );
	    cvResetImageROI( img_src );
	    
	    if( USE_KINECT_DEP ){
	      cvSetImageROI( dep_src, bounding_rect );
	      dep_crop = cvCreateImage( cvSize( bounding_rect.width, bounding_rect.height ),
					IPL_DEPTH_16U, 1 );
	      cvCopy( dep_src, dep_crop, NULL );
	      cvResetImageROI( dep_src );

	      top_left[0] = bounding_rect.x; top_left[1] = bounding_rect.y;
	    }
	    
	  } else {
	    cv::Rect bounding_rect( (int)(30*frame_ratio), (int)(10*frame_ratio),
				    (int)(100*frame_ratio), (int)(100*frame_ratio) );
	    cvSetImageROI( img_src, bounding_rect );
	    img_crop=cvCreateImage( cvSize(bounding_rect.width,bounding_rect.height), IPL_DEPTH_8U, 3 );
	    cvCopy( img_src, img_crop, NULL );
	    cvResetImageROI( img_src );
	    //img_crop=img_src;

	    if( USE_KINECT_DEP ){
	      cvSetImageROI( dep_src, bounding_rect );
	      dep_crop = cvCreateImage( cvSize( bounding_rect.width, bounding_rect.height ),
					IPL_DEPTH_16U, 1 );
	      cvCopy( dep_src, dep_crop, NULL );
	      cvResetImageROI( dep_src );

	      top_left[0] = bounding_rect.x; top_left[1] = bounding_rect.y;
	    }
	    
	  }
	  
	  if (img_crop != NULL)
	    {
	      bool result;
	      if( MODEL_TYPE == 0 || MODEL_TYPE == 2 )
		result = kdm->Process(imfea2, img_crop);
	      else if( MODEL_TYPE == 3 || MODEL_TYPE == 4 ){
		if( MODEL_TYPE == 4 )
		  result = kdm->Process(imfea2, dep_crop, top_left);
		else
		  result = kdm->Process(imfea2, dep_crop);
	      }
	      //string object_name = kdm->GetObjectName(imfea2);
	      object_name = kdm->GetObjectName(imfea2);
	      cvPutText(img_src, object_name.c_str(), cvPoint(10,20), &font,cvScalar(0,256,0));
	      
	      cvShowImage("Cropped Frame", img_crop);
	      cvShowImage("Processed Frame", img_src);
	      if( USE_KINECT_DEP ){
		cv::Mat tmp( dep_src ); cv::Mat depthshow;
		tmp.convertTo(depthshow, CV_8U, 256.0/4096.0, 0);
		imshow("Depth Frame", depthshow);
		//cvShowImage("Depth Frame", dep_src);
	      }
	      //cv::DisplayOverlay("Camera View", object_name.c_str(), 1000);
	      cvReleaseImage(&img_crop);
	    }
	}
      
      key = cvWaitKey(1);
    }
  delete kdm;
  cout << "Descriptor thread ended..." << endl;
  
}

void GetFileListFromDirectory(vector<string>& src, const char* directory)
{
  if (boost::filesystem::is_directory(directory))
    {
      for (boost::filesystem::directory_iterator itr(directory); itr != boost::filesystem::directory_iterator(); ++itr)
	{
	  if (!is_directory(itr->status()))
	    {
#ifdef NEW_BOOST
	      string fn = itr->path().filename().string(); // new boost version
#else
	      string fn = itr->path().filename(); // old boost version?
#endif
	      src.push_back(fn);
	      
	    }
	  
	}
      
    } else {
    cout << "Image directory not found in path!" << endl;
  }
}

void DataSetDemo()
{
  // font stuff
  SetupFont(font);
  
  vector<string> filelist;
  GetFileListFromDirectory(filelist,"testim");
  cout << "first name: " << filelist[0] << endl;
  cvNamedWindow("Processed Image", CV_WINDOW_AUTOSIZE);
  KernelDescManager* kdm = new KernelDescManager(string(model_name), string(model_file),
						 string(model_var), string(param_file));
  
  vector<string>::iterator itr;
  for (itr = filelist.begin(); itr < filelist.end();++itr)
    {
      MatrixXf imfea2;
      VectorXf top_left(2);
      IplImage* img_init;
      if( MODEL_TYPE == 0 || MODEL_TYPE == 2 )
	img_init = cvLoadImage( string("./testim/" + *itr).c_str(), CV_LOAD_IMAGE_ANYCOLOR);
      else
	img_init = cvLoadImage( string("./testim/" + *itr).c_str(), CV_LOAD_IMAGE_ANYDEPTH );
      bool result;
      if( MODEL_TYPE == 0 || MODEL_TYPE == 2 )
	result = kdm->Process(imfea2,img_init);
      else if( MODEL_TYPE == 3 )
	result = kdm->Process(imfea2,img_init);
      else if( MODEL_TYPE == 4 ){
	top_left[0] = 317; top_left[1] = 169;//TODO:Load loc.txt file.
	result = kdm->Process(imfea2,img_init, top_left);
      }
      //cvShowImage("ProcessedImage", frame);
      string object_name = kdm->GetObjectName(imfea2);
      std::cout << "(ImagePath:" << string(*itr).c_str() << ")" << std::endl << std::endl;
      cvPutText(img_init, object_name.c_str(), cvPoint(30,30), &font ,cvScalar(0,256,0));
      if( !run_debug_mode ){
	cvShowImage("Processed Image", img_init);
	char c = cvWaitKey(0);
      }
      cvReleaseImage(&img_init);
    }	
}

int main(int argc, char* argv[]) {
  
  int opt = -1;
  while (opt != 1 || opt != 2 || opt != 3)
    {
      cout << "Enter '1' for camera demo (full image)," << endl
	   << "      '2' for camera demo (w/ GrabCut)," << endl
	   << "      '3' for demo with image dataset,"   << endl
	   << "      '4' for debug with image dataset,"  << endl
	   << "      '5' for camera demo (self cut)"    << endl;
      cin >> opt;
      cout << "Opt: " << opt << endl;
      if (opt==1 || opt==2 || opt==5)
	{
	  
	  // On Windows: change camera resolution only works in the main thread
	  capture = cvCaptureFromCAM(0);
	  if (!capture)
	    {
	      fprintf(stderr, "Cannot open the webcam.\n");
	      return 1;
	    }
	  cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, FRAME_WIDTH_LIVE );
	  cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT_LIVE );
	  
	  int key = 0;
	  if (opt==2) {
	    run_grab_cut=true;
	  }
	  if (opt==5) {
	    run_self_cut=true;
	  }
	  boost::thread workerThread(&ThreadCaptureFrame);
	  boost::thread descriptorThread(&ThreadRunDescriptors);
	  cout << "Starting worker thread" << endl;
	  workerThread.join();
	  cout << "Starting descriptor thread" << endl;
	  descriptorThread.join();
	} else if (opt == 3 || opt == 4)
	{
	  if (opt == 4 )
	    run_debug_mode = true;
	  DataSetDemo();
	} else 
	{
	  ClearCinBufferFlags();
	}
    }
  fflush(stdout);
  
  return 0;
}
