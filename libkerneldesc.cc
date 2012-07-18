#include "libkerneldesc.h"
#include <limits.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include "liblinear-dense-float/linear.h"

KernelDescManager::KernelDescManager(string model_name,string model_file, string model_var, string param_file,
				     unsigned int MODEL_TYPE, int MAX_IMAGE_SIZE)
  : model_name(model_name), model_file(model_file), model_var(model_var), param_file(param_file),
    MODEL_TYPE(MODEL_TYPE), MAX_IMAGE_SIZE(MAX_IMAGE_SIZE)
{
  this->model_dir = string("./model_my/") + model_file +string(".mat");
  cout << "Model Dir: " << this->model_dir.c_str() << " Model file: " << model_file <<  endl;
  this->param_dir = string("./kdes/") + param_file + string(".mat");
  this->model_kdes = load_mat("Model file",model_dir.c_str());
  this->kdes_params = load_mat("Param file", param_dir.c_str());
  this->model_kdes_treeroot = model_var + string("->svm->classname");
  this->model_list = get_charlist(this->model_kdes, this->model_kdes_treeroot.c_str());
  //PrintModelList();
  if (this->model_list == NULL)
    printf("WARNING: the model list is nulli\n");
  else 
    printf("This is a kernel descriptor demo for %d-class object recognition\n",(int)this->model_list->size());
  
}

int KernelDescManager::GetObjectName_liblinear(MatrixXf& imfea)
{
  /*
   *
   * Use liblinear program
   *
   */
  //std::cout << imfea_s.cols() << " " << imfea_s.rows() << std::endl;
  //std::cout << imfea_s( 100,0 ) << "test" << std::endl;//OK!! imfea_s( 14000-1, 0 )
  
  struct model* linearmodel;
  if( ( linearmodel = load_model( "./model_my/linearmodel.linear" ) ) == 0 ){
    std::cerr << "Can not open model file" << std::endl;
    return -1;
  }
  int nr_class = get_nr_class( linearmodel );
  int nr_feature = get_nr_feature( linearmodel );
  struct feature_node *x;
  x = (struct feature_node *) malloc( 150000 * sizeof(struct feature_node) );
  int i;
  for( i = 0; i < imfea.rows(); i++ ){
    x[i].index = i+1;
    x[i].value = imfea( i, 0 );
  }
  x[i].index = -1;
  if( linearmodel->bias >= 0 ){
    std::cerr << "model->bias >= 0! Not supported!!" << std::endl;
  }
  int predict_label = predict( linearmodel, x );
    
  free(x);
  destroy_model( linearmodel );
  return predict_label;
}

string KernelDescManager::GetObjectName(MatrixXf& imfea)
{
  std::cout << "MODELTYPE:" << MODEL_TYPE << std::endl;
  
  MatrixXf svmW;
  if( MODEL_TYPE == 0 ){
    get_matrix(svmW, this->model_kdes,"modelgkdes->svm->w");
  }else if( MODEL_TYPE == 2 ){
    get_matrix(svmW, this->model_kdes,"modelrgbkdes->svm->w");
  }
  MatrixXf svmMinValue;
  MatrixXf svmMaxValue;
  if( MODEL_TYPE == 0 ){
    get_matrix(svmMaxValue, this->model_kdes,"modelgkdes->svm->maxvalue"); 
    get_matrix(svmMinValue, this->model_kdes,"modelgkdes->svm->minvalue");
	}else if( MODEL_TYPE == 2 ){
    get_matrix(svmMaxValue, this->model_kdes,"modelrgbkdes->svm->maxvalue"); 
    get_matrix(svmMinValue, this->model_kdes,"modelrgbkdes->svm->minvalue");
  }
  MatrixXf imfea_s = scaletest_linear( imfea, svmMinValue, svmMaxValue);
  
  int predict_label = this->GetObjectName_liblinear( imfea_s );
  if( predict_label == -1 )
    return string("ERROR");
  string str_result = (* this->model_list)[predict_label-1];
  std::cout << "predict_label:" << predict_label << "  "
	    << "predict_name:" << str_result << std::endl;
  return str_result;
  
  //Original Old
  /*
  MatrixXf res = ( imfea_s.transpose()*(svmW).transpose() ).transpose();
  float max_result = -FLT_MAX;
  int max_index = 0;
  for (int i = 0; i < res.rows(); i++)
    {
      if (res(i,0) > max_result)
	{
	  max_result = res(i,0);
	  max_index = i;
	} 
      //cout << (* this->model_list)[i] << " " << res(i,0) << endl;
    }	
  //string str_result =(* this->model_list)[max_index];
  cout << "Item is: " <<  str_result << endl;
  */
  
  //Original New
  /*
    vector<mypair> vec;
    for(int i=0; i<res.rows(); i++) {
    vec.push_back( mypair( res(i,0), myindex(i,0) ) );
    }
    sort( vec.begin(), vec.end(), comparator );
    vector<mypair>::iterator itr=vec.begin();
    cout << showpos;
    cout << "OBJECT IS: " << setw(12) << (* this->model_list)[itr->second.first] << "    ";
    string str_result =(* this->model_list)[itr->second.first];
    for(int k=1; k<min(OBJECTS_TO_PRINT,(int)res.rows()); k++) {
    itr++;
    cout << "[" << setw(12) << (* this->model_list)[itr->second.first] << "] ";
    }
    itr=vec.begin();
    for(int k=0; k<min(OBJECTS_TO_PRINT,(int)res.rows()); k++) {
    cout << " " << setw(12) << (float)itr->first << " ";
    itr++;
    }
    cout << "\r";
    //cout << endl;
    return str_result;
    */
  
}

/*
  bool KernelDescManager::Process(MatrixXf& imfea, string image_path)
  {
  printf("Image path: %s\n", image_path.c_str());
  IplImage* img_init = cvLoadImage(image_path.c_str(),CV_LOAD_IMAGE_COLOR);
  
  //printf("Final image '%s' width: %d height: %d\n", image_path.c_str(), img->width, img->height);
  Process(imfea, img_init);
  cvReleaseImage(&img_init);
  
  }
*/
bool KernelDescManager::Process(MatrixXf&imfea, IplImage* image, const VectorXf& top_left)
{
  this->top_left=top_left;
  return Process(imfea,image);
}

bool KernelDescManager::Process(MatrixXf&imfea, IplImage* image)
{
  if ( !(MODEL_TYPE==0 || MODEL_TYPE==2 || MODEL_TYPE==3 || MODEL_TYPE==4) ) {
    printf("MODEL_TYPE %d unsupported.\n",MODEL_TYPE);
    return false;
  }
  // do the first object in the list
  // convert to floats
  IplImage* img_init = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_32F, image->nChannels);
  if (!img_init)
    printf("Image is unavailable!\n");
  //cvConvertScale(image, img_init,1.0/255, 0);
  switch (MODEL_TYPE) {
  case 3 :
  case 4 :
    assert( image->nChannels==1 );  // must be grayscale
    cvConvertScale(image, img_init,1.0/1000, 0);
    break;
  case 0 :
  case 1 :
  case 2 :
  default :
    cvConvertScale(image, img_init,1.0/255, 0);
    break;
  }
  //matvarplus_t* modelkdes = load_mat(this->model_file.c_str(), (this->model_file+".mat").c_str());
  
  IplImage* img;
  
  /*
   *
   *Matlab's program don't use resize function.
   *
   */
  
  /*
    //const double EPS_RATIO=0.0001;
    //int max_imsize=(int)get_value<float>(this->model_kdes, (model_var+"->kdes->max_imsize").c_str() );
    //int min_imsize=(int)get_value<float>(this->model_kdes, (model_var+"->kdes->min_imsize").c_str() );
    double ratio, ratio_f, ratio_max, ratio_min;
    ratio_f=1.0;
    if (MAX_IMAGE_SIZE>0) {
    ratio_f = max( ratio_f, max( (double)img_init->width/this->MAX_IMAGE_SIZE, (double)img_init->height/this->MAX_IMAGE_SIZE ) );
    }
    ratio_max = max( max( (double)img_init->width/max_imsize, (double)img_init->height/max_imsize ), 1.0 );
    ratio_min = min( min( (double)img_init->width/min_imsize, (double)img_init->height/min_imsize ), 1.0 );
    if (ratio_min<1.0-EPS_RATIO) {
    ratio=ratio_min;
    } else {
    ratio=max(ratio_f,ratio_max);
    }
    if (ratio>1.0-EPS_RATIO || ratio<1.0-EPS_RATIO) {
    int method=CV_INTER_CUBIC;
    if (MODEL_TYPE==4 || MODEL_TYPE==5) method=CV_INTER_NN;   // nearest neighbor for depth image
    img = cvCreateImage( cvSize((img_init->width)/ratio,(img_init->height)/ratio), IPL_DEPTH_32F, img_init->nChannels);
    cvResize( img_init, img, method );
    cvReleaseImage(&img_init);
    } else {
    img=img_init;
    }
  */
  
  img = cvCreateImage( cvSize( img_init->width, img_init->height ), IPL_DEPTH_32F, img_init->nChannels );
  img = img_init;
  int img_w=img->width, img_h=img->height;
  
  //void GKDESDense(IplImage* im, matvarplus_t* kdes_params, float grid_space, float patch_size, float low_contrast) {
  
  //cout << "Start KDES computation..." << "(" << img_w << " " << img_h << ")" << endl;
  Timer timer;
  double exec_time, exec_time2;
  timer.start();
  MatrixXf feaArr, feaMag, fgrid_y, fgrid_x;
  if (MODEL_TYPE==0 || MODEL_TYPE==3) {
    //cvCvtColor(im, im_temp, CV_RGB2GRAY);
    GKDESDense(feaArr, feaMag, fgrid_y, fgrid_x, img, this->kdes_params, get_value<float>(this->model_kdes, "modelgkdes->kdes->grid_space"),
	       get_value<float>(this->model_kdes, "modelgkdes->kdes->patch_size"),
	       get_value<float>(this->model_kdes,"modelgkdes->kdes->low_contrast"));
  }
  if (MODEL_TYPE==2) {
    RGBKDESDense(feaArr, feaMag, fgrid_y, fgrid_x, img, this->kdes_params, get_value<float>(this->model_kdes, "modelrgbkdes->kdes->grid_space"),
		 get_value<float>(this->model_kdes, "modelrgbkdes->kdes->patch_size"));
  }
  if (MODEL_TYPE==4) {
    SpinKDESDense(feaArr, fgrid_y, fgrid_x, img, this->top_left, this->kdes_params, get_value<float>(this->model_kdes, "modelspinkdes->kdes->grid_space"),
		  get_value<float>(this->model_kdes, "modelspinkdes->kdes->patch_size"));
  }
  exec_time = timer.get();
  //cout << "KDES Execution time... " << exec_time << endl;
  
  timer.start();
  //MatrixXf imfea;
  MatrixXf emkWords;
  get_matrix(emkWords, this->model_kdes, (string(this->model_var)+"->emk->words").c_str());
  MatrixXf emkG;
  get_matrix(emkG, this->model_kdes, (string(this->model_var)+"->emk->G").c_str());
  MatrixXf emkPyramid;
  get_matrix(emkPyramid, this->model_kdes, (string(this->model_var)+"->emk->pyramid").c_str());
  CKSVDEMK(imfea, feaArr, feaMag, fgrid_y, fgrid_x, img_h, img_w,
	   emkWords,
	   emkG,
	   emkPyramid,
	   get_value<float>(this->model_kdes, (string(this->model_var)+"->emk->kparam").c_str()) );
  exec_time2 = timer.get();
  //cout << "EMK Execution time... " << exec_time2 << endl;
  
  //cout << "Total Execution time... " << exec_time+exec_time2 << endl;
  cout << "Execution time: " << setw(8) << exec_time+exec_time2 << "    ";
  //printf("end\n");
  cvReleaseImage(&img);
  //if (img)
  //cvReleaseImage(&img);
  //if (modelkdes)
  //delete modelkdes;
  
  fflush(stdout);
  return true;	
}

void KernelDescManager::PrintModelList()
{
  printf("Available models:\n");
  foreach(string s, *model_list) {
    printf("\t%s\n", s.c_str());
  }
  
}

KernelDescManager::~KernelDescManager()
{
  delete model_kdes;
  delete kdes_params;
  delete model_list;
}
