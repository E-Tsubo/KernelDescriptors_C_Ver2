#include "libkerneldesc.h"
#include <limits.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include "liblinear-dense-float/linear.h"

KernelDescManager::KernelDescManager(string model_name,string model_file, string model_var, string param_file,
				     unsigned int MODEL_TYPE, int MAX_IMAGE_SIZE)
  : model_name(model_name), model_file(model_file), model_var(model_var), param_file(param_file),
    MODEL_TYPE(MODEL_TYPE), MAX_IMAGE_SIZE(MAX_IMAGE_SIZE)
{
  if( USE_COMBINE_MODEL != 1 ){
    this->MODEL_TYPE = USE_MODEL_TYPE;
    this->model_dir = string("./model_my/") + model_file +string(".mat");
    cout << "Model Dir: " << this->model_dir.c_str() << " Model file: " << model_file <<  endl;
    this->param_dir = string("./kdes/") + param_file + string(".mat");
    this->model_kdes = load_mat("Model file",model_dir.c_str());
    this->kdes_params = load_mat("Param file", param_dir.c_str());
    this->model_kdes_treeroot = model_var + string("->svm->classname");
    this->model_list = get_charlist(this->model_kdes, this->model_kdes_treeroot.c_str());
    this->linearmodel_file = string("./model_my/") + model_file + string(".linear");
    PrintModelList();
    if (this->model_list == NULL)
      printf("WARNING: the model list is nulli\n");
    else 
      printf("This is a kernel descriptor demo for %d-class object recognition\n",(int)this->model_list->size());
  }else{
    this->MODEL_TYPE = USE_MODEL_TYPE;
    
    this->comMODEL_TYPE = 0;
    this->USE_TYPE_COUNT = 0;
    for( int i = 0; i < MAX_COMBINE_MODEL_TYPE; i++ ){
      if( MODEL_FLAG[i] == 1 ){
	this->comMODEL_TYPE += i * i;
	this->USE_TYPE_COUNT++;
	
	this->commodel_dir[i] = string("./model_my/") + model_files[i] +string(".mat");
	cout << "Model Dir: " << this->commodel_dir[i].c_str() << " Model file: " << model_files[i] <<  endl;
	this->comparam_dir[i] = string("./kdes/") + param_files[i] + string(".mat");
	this->commodel_kdes[i] = load_mat("Model file",commodel_dir[i].c_str());
	this->comkdes_params[i] = load_mat("Param file", comparam_dir[i].c_str());
	this->commodel_kdes_treeroot = model_vars[i] + string("->svm->classname");
	this->commodel_list = get_charlist(this->commodel_kdes[i], this->commodel_kdes_treeroot.c_str());
	
      }
    }
    
    this->finalmodel_kdes = load_mat("Model file", "./model_my/combinekdes.mat");
    //this->comlinearmodel_file = string("./model_my/") + string("combine") + string(".linear");
    this->linearmodel_file = string("./model_my/") + string("combinekdes") + string(".linear");
    
    PrintModelList();
    if( this->commodel_list == NULL )
      printf("WARNING: the model list is nulli\n");
    else 
      printf("This is a kernel descriptor demo for %d-class object recognition\n",(int)this->commodel_list->size());
    
  }
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
  if( ( linearmodel = load_model( this->linearmodel_file.c_str() ) ) == 0 ){
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
  if( MODEL_TYPE == 0 || MODEL_TYPE == 3 ){
    get_matrix(svmW, this->model_kdes,"modelgkdes->svm->w");
  }else if( MODEL_TYPE == 2 ){
    get_matrix(svmW, this->model_kdes,"modelrgbkdes->svm->w");
  }else if( MODEL_TYPE == 4 ){
    get_matrix(svmW, this->model_kdes,"modelspinkdes->svm->w");
  }
  MatrixXf svmMinValue;
  MatrixXf svmMaxValue;
  if( MODEL_TYPE == 0 || MODEL_TYPE == 3 ){
    get_matrix(svmMaxValue, this->model_kdes,"modelgkdes->svm->maxvalue"); 
    get_matrix(svmMinValue, this->model_kdes,"modelgkdes->svm->minvalue");
  }else if( MODEL_TYPE == 2 ){
    get_matrix(svmMaxValue, this->model_kdes,"modelrgbkdes->svm->maxvalue"); 
    get_matrix(svmMinValue, this->model_kdes,"modelrgbkdes->svm->minvalue");
  }else if( MODEL_TYPE == 4 ){
    get_matrix(svmMaxValue, this->model_kdes,"modelspinkdes->svm->maxvalue"); 
    get_matrix(svmMinValue, this->model_kdes,"modelspinkdes->svm->minvalue");
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

string KernelDescManager::GetObjectNameCombine(MatrixXf& imfea)
{
  std::cout << "MODEL TYPE:Combine MODEL" << std::endl;

  MatrixXf svmW;
  get_matrix(svmW, this->finalmodel_kdes,"combinekdes->svm->w");
  
  MatrixXf svmMinValue;
  MatrixXf svmMaxValue;
  get_matrix(svmMaxValue, this->finalmodel_kdes,"combinekdes->svm->maxvalue"); 
  get_matrix(svmMinValue, this->finalmodel_kdes,"combinekdes->svm->minvalue");
  
  MatrixXf imfea_s = scaletest_linear( imfea, svmMinValue, svmMaxValue);
  
  int predict_label = this->GetObjectName_liblinear( imfea_s );
  if( predict_label == -1 )
    return string("ERROR");
  string str_result = (* this->commodel_list)[predict_label-1];
  std::cout << "predict_label:" << predict_label << "  "
	    << "predict_name:" << str_result << std::endl;
  return str_result;
  
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
    assert( image->nChannels==1 );//must be grayscale(depth image)
    cvConvertScale(image, img_init,1.0/1000, 0);
    break;
  case 0 :
  case 1 :
  case 2 :
  default :
    cvConvertScale(image, img_init,1.0/255, 0);//normalized
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
    //In GKDESDense func, image is converted from RGB to Gray.
    //So, input image is color image.
    GKDESDense(feaArr, feaMag, fgrid_y, fgrid_x, img, this->kdes_params,
	       get_value<float>(this->model_kdes, "modelgkdes->kdes->grid_space"),
	       get_value<float>(this->model_kdes, "modelgkdes->kdes->patch_size"),
	       get_value<float>(this->model_kdes, "modelgkdes->kdes->low_contrast"));
  }
  if (MODEL_TYPE==2) {
    RGBKDESDense(feaArr, feaMag, fgrid_y, fgrid_x, img, this->kdes_params,
		 get_value<float>(this->model_kdes, "modelrgbkdes->kdes->grid_space"),
		 get_value<float>(this->model_kdes, "modelrgbkdes->kdes->patch_size"));
  }
  if (MODEL_TYPE==4) {
    SpinKDESDense(feaArr, fgrid_y, fgrid_x, img, this->top_left, this->kdes_params,
		  get_value<float>(this->model_kdes, "modelspinkdes->kdes->grid_space"),
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

bool KernelDescManager::ProcessCombine(MatrixXf&imfea, IplImage* image, IplImage* depimage, const VectorXf& top_left)
{
  this->comtop_left=top_left;
  
  if ( !(MODEL_TYPE==0 || MODEL_TYPE==2 || MODEL_TYPE==3 || MODEL_TYPE==4) ) {
    printf("MODEL_TYPE %d unsupported.\n",MODEL_TYPE);
    return false;
  }
  // do the first object in the list
  // convert to floats
  IplImage* img_init = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_32F, image->nChannels);
  IplImage* dep_init = cvCreateImage(cvSize(depimage->width, depimage->height), IPL_DEPTH_32F, depimage->nChannels);
  if (!img_init && !dep_init)
    printf("Image is unavailable!\n");
  
  switch (MODEL_TYPE) {
  case 3 ://For Depth KDES
  case 4 :
    assert( depimage->nChannels==1 );//must be grayscale(depth image)
    cvConvertScale(depimage, dep_init,1.0/1000, 0);
    break;
  case 0 ://For RGB KDES
  case 1 :
  case 2 :
  default :
    cvConvertScale(image, img_init,1.0/255, 0);//normalized
    break;
  }
  
  IplImage* img, *dep;
  img = cvCreateImage( cvSize( img_init->width, img_init->height ), IPL_DEPTH_32F, img_init->nChannels );
  dep = cvCreateImage( cvSize( dep_init->width, dep_init->height ), IPL_DEPTH_32F, dep_init->nChannels );
  img = img_init;
  dep = dep_init;
  int img_w=img->width, img_h=img->height;
  int dep_w=dep->width, dep_h=dep->height;
  
  Timer timer;
  double exec_time, exec_time2;
  timer.start();
  
  MatrixXf feaArr[MAX_COMBINE_MODEL_TYPE];
  
  boost::thread* th[MAX_COMBINE_MODEL_TYPE];
  for( int i = 0; i < MAX_COMBINE_MODEL_TYPE; i++ ){
    if( MODEL_FLAG[i] == 1 )
      th[i] = new boost::thread( boost::bind( &KernelDescManager::threadKDES, this, &feaArr[i], img, dep, i ) );
  }
  
  for( int i = 0; i < MAX_COMBINE_MODEL_TYPE; i++ ){
    if( MODEL_FLAG[i] == 1 )
      th[i]->join();
  }
  
  for( int i = 0; i < MAX_COMBINE_MODEL_TYPE; i++ ){
    if( MODEL_FLAG[i] == 1 )
      delete th[i];
  }
  
  //統合フェーズ imfea <--kakunou
  int f_row = 0;
  for( int i = 0; i < MAX_COMBINE_MODEL_TYPE; i++ ){
    if( MODEL_FLAG[i] == 1 ){
      f_row += feaArr[i].rows();
    }
  }
  MatrixXf combine(f_row, 1);
  int index = 0;
  for( int i = 0; i < MAX_COMBINE_MODEL_TYPE; i++ ){
    if( MODEL_FLAG[i] == 1 ){
      
      for( int j = 0; j < feaArr[i].rows(); j++ ){
	combine( index, 0 ) = feaArr[i]( j, 0 );
	index++;
      }
      
    }
  }
  imfea = combine;
  
  exec_time = timer.get();
  cout << "Execution time: " << setw(8) << exec_time << "    ";
  
  cvReleaseImage(&img);
  cvReleaseImage(&dep);

  fflush(stdout);
  return true;
  
  /*
  for( int i = 0; i < MAX_COMBINE_MODEL_TYPE; i++ ){
    if( MODEL_FLAG[i] == 1 ){
      
      if( i == 0 ){
	boost::thread th0( boost::bind( &KernelDescManager::threadKDES, this, feaArr[i], img, dep, i ) );
	th0.join();
      }
      if( i == 1 ){
	boost::thread th1( boost::bind( &KernelDescManager::threadKDES, this, feaArr[i], img, dep, i ) );
	th1.join();//not supported now.
      }
      if( i == 2 ){
	boost::thread th2( boost::bind( &KernelDescManager::threadKDES, this, feaArr[i], img, dep, i ) );
	th2.join();
      }
      if( i == 3 ){
	boost::thread th3( boost::bind( &KernelDescManager::threadKDES, this, feaArr[i], img, dep, i ) );
	th3.join();
      }
      if( i == 4 ){
	boost::thread th4( boost::bind( &KernelDescManager::threadKDES, this, feaArr[i], img, dep, i ) );
	th4.join();
      }

      
    }
  }
  */
  /*
  //Thread of KDES
  if( MODEL_FLAG[0] == 1 )
    boost::thread thr_GKDES( boost::bind(&GKDESDense, feaArr[0], feaMag[0], fgrid_y[0], fgrid_x[0],
					 img, this->comkdes_params[0],
					 get_value<float>(this->commodel_kdes[0], "modelgkdes->kdes->grid_space"),
					 get_value<float>(this->commodel_kdes[0], "modelgkdes->kdes->patch_size"),
					 get_value<float>(this->commodel_kdes[0], "modelgkdes->kdes->low_contrast") ) );
  
  if( MODEL_FLAG[2] == 1 )
    boost::thread thr_RGBKDES( boost::bind(&RGBKDESDense, feaArr[2], feaMag[2], fgrid_y[2], fgrid_x[2],
					   img, this->comkdes_params[2],
					   get_value<float>(this->commodel_kdes[2], "modelrgbkdes->kdes->grid_space"),
					   get_value<float>(this->commodel_kdes[2], "modelrgbkdes->kdes->patch_size") ) );

  if( MODEL_FLAG[3] == 1 )
    boost::thread thr_DGKDES( boost::bind(&GKDESDense, feaArr[3], feaMag[3], fgrid_y[3], fgrid_x[3],
					  dep, this->comkdes_params[3],
					  get_value<float>(this->commodel_kdes[3], "modelgkdes->kdes->grid_space"),
					  get_value<float>(this->commodel_kdes[3], "modelgkdes->kdes->patch_size"),
					  get_value<float>(this->commodel_kdes[3], "modelgkdes->kdes->low_contrast") ) );
  
  if( MODEL_FLAG[4] == 1 )
    boost::thread thr_SPINKDES( boost::bind(&SpinKDESDense, feaArr[4], fgrid_y[4], fgrid_x[4], dep,
					    this->comtop_left, this->comkdes_params[4],
					    get_value<float>(this->commodel_kdes[4], "modelspinkdes->kdes->grid_space"),
					    get_value<float>(this->commodel_kdes[4], "modelspinkdes->kdes->patch_size"),
					    (double)0.01, (double)5.0 ) );
  */
  
}


void KernelDescManager::threadKDES( MatrixXf *fea, IplImage* img, IplImage* dep, int flag )
{
  //MatrixXf feaArr, feaMag, fgrid_y, fgrid_x;
  if (flag==0) {
    std::cout << "GKDES 0, start..." << std::endl; fflush(stdout);
    MatrixXf feaArr, feaMag, fgrid_y, fgrid_x;
    int grid_space, patch_size; float low_contrast;
    {
      boost::mutex::scoped_lock lock(mutex);
      grid_space = (int)get_value<float>(this->commodel_kdes[0], "modelgkdes->kdes->grid_space");
      patch_size = (int)get_value<float>(this->commodel_kdes[0], "modelgkdes->kdes->patch_size");
      low_contrast = get_value<float>(this->commodel_kdes[0], "modelgkdes->kdes->low_contrast");
    }
    GKDESDense(feaArr, feaMag, fgrid_y, fgrid_x, img, this->comkdes_params[0],
	       grid_space,
	       patch_size,
	       low_contrast);
    
    MatrixXf emkWords;
    get_matrix(emkWords, this->commodel_kdes[0], (string(model_vars[0])+"->emk->words").c_str());
    MatrixXf emkG;
    get_matrix(emkG, this->commodel_kdes[0], (string(model_vars[0])+"->emk->G").c_str());
    MatrixXf emkPyramid;
    get_matrix(emkPyramid, this->commodel_kdes[0], (string(model_vars[0])+"->emk->pyramid").c_str());
    CKSVDEMK(*fea, feaArr, feaMag, fgrid_y, fgrid_x, img->height, img->width,
	     emkWords,
	     emkG,
	     emkPyramid,
	     get_value<float>(this->commodel_kdes[0], (string(model_vars[0])+"->emk->kparam").c_str()) );
    std::cout << "GKDES 0, End now." << std::endl; fflush(stdout);
    return;
  }else if (flag==2) {
    std::cout << "RGBKDES 2, start..." << std::endl; fflush(stdout);
    MatrixXf feaArr, feaMag, fgrid_y, fgrid_x;
    int grid_space, patch_size;
    {
      boost::mutex::scoped_lock lock(mutex);
      grid_space = (int)get_value<float>(this->commodel_kdes[2], "modelrgbkdes->kdes->grid_space");
      patch_size = (int)get_value<float>(this->commodel_kdes[2], "modelrgbkdes->kdes->patch_size");
    }
    RGBKDESDense(feaArr, feaMag, fgrid_y, fgrid_x, img, this->comkdes_params[2],
		 grid_space, patch_size );
    
    MatrixXf emkWords;
    get_matrix(emkWords, this->commodel_kdes[2], (string(model_vars[2])+"->emk->words").c_str());
    MatrixXf emkG;
    get_matrix(emkG, this->commodel_kdes[2], (string(model_vars[2])+"->emk->G").c_str());
    MatrixXf emkPyramid;
    get_matrix(emkPyramid, this->commodel_kdes[2], (string(model_vars[2])+"->emk->pyramid").c_str());
    CKSVDEMK(*fea, feaArr, feaMag, fgrid_y, fgrid_x, img->height, img->width,
	     emkWords,
	     emkG,
	     emkPyramid,
	     get_value<float>(this->commodel_kdes[2], (string(model_vars[2])+"->emk->kparam").c_str()) );
    std::cout << "RGBKDES 2, End now." << std::endl; fflush(stdout);
    return;
  }else if (flag==3) {
    std::cout << "GKDES 3, start..." << std::endl; fflush(stdout);
    MatrixXf feaArr, feaMag, fgrid_y, fgrid_x;
    int grid_space, patch_size; float low_contrast;
    {
      boost::mutex::scoped_lock lock(mutex);
      grid_space = (int)get_value<float>(this->commodel_kdes[3], "modelgkdes->kdes->grid_space");
      patch_size = (int)get_value<float>(this->commodel_kdes[3], "modelgkdes->kdes->patch_size");
      low_contrast = get_value<float>(this->commodel_kdes[3], "modelgkdes->kdes->low_contrast");
    }
    GKDESDense(feaArr, feaMag, fgrid_y, fgrid_x, dep, this->comkdes_params[3],
	       grid_space,
	       patch_size,
	       low_contrast);
    
    MatrixXf emkWords;
    get_matrix(emkWords, this->commodel_kdes[3], (string(model_vars[3])+"->emk->words").c_str());
    MatrixXf emkG;
    get_matrix(emkG, this->commodel_kdes[3], (string(model_vars[3])+"->emk->G").c_str());
    MatrixXf emkPyramid;
    get_matrix(emkPyramid, this->commodel_kdes[3], (string(model_vars[3])+"->emk->pyramid").c_str());
    CKSVDEMK(*fea, feaArr, feaMag, fgrid_y, fgrid_x, dep->height, dep->width,
	     emkWords,
	     emkG,
	     emkPyramid,
	     get_value<float>(this->commodel_kdes[3], (string(model_vars[3])+"->emk->kparam").c_str()) );
    std::cout << "GKDES 3, End now." << std::endl; fflush(stdout);
    return;
  }else if (flag==4) {
    std::cout << "SpinKDES 4, start..." << std::endl; fflush(stdout);
    MatrixXf feaArr, feaMag, fgrid_y, fgrid_x;
    int grid_space, patch_size;
    {
      boost::mutex::scoped_lock lock(mutex);
      grid_space = (int)get_value<float>(this->commodel_kdes[4], "modelspinkdes->kdes->grid_space");
      patch_size = (int)get_value<float>(this->commodel_kdes[4], "modelspinkdes->kdes->patch_size");
    }
    SpinKDESDense(feaArr, fgrid_y, fgrid_x, dep, this->comtop_left, this->comkdes_params[4],
		  grid_space, patch_size );
    
    MatrixXf emkWords;
    get_matrix(emkWords, this->commodel_kdes[4], (string(model_vars[4])+"->emk->words").c_str());
    MatrixXf emkG;
    get_matrix(emkG, this->commodel_kdes[4], (string(model_vars[4])+"->emk->G").c_str());
    MatrixXf emkPyramid;
    get_matrix(emkPyramid, this->commodel_kdes[4], (string(model_vars[4])+"->emk->pyramid").c_str());
    CKSVDEMK(*fea, feaArr, feaMag, fgrid_y, fgrid_x, dep->height, dep->width,
	     emkWords,
	     emkG,
	     emkPyramid,
	     get_value<float>(this->commodel_kdes[4], (string(model_vars[4])+"->emk->kparam").c_str()) );
    std::cout << "SpinKDES 4, End now." << std::endl; fflush(stdout);
    return;
  }
  return;
}

void KernelDescManager::PrintModelList()
{
  if( USE_COMBINE_MODEL != 1 ){
    printf("Available models:\n");
    foreach(string s, *model_list) {
      printf("\t%s\n", s.c_str());
    }
  }else{
    printf("Available models:\n");
    foreach(string s, *commodel_list) {
      printf("\t%s\n", s.c_str());
    }
  }
}

KernelDescManager::~KernelDescManager()
{
  delete model_kdes;
  delete kdes_params;
  delete model_list;
}
