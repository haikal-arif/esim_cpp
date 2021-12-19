#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <fmt/core.h>

#include "esim"

template<class T>
class SafeQueue {

public:
	SafeQueue() :
		_queue(), _mu(), _cond(){}

	~SafeQueue(){}
	
	void push(T& input_item) {
		std::unique_lock<std::mutex> locker(_mu);
		_queue.push(input_item);
		_cond.notify_one();
	}

	T front() {
		std::unique_lock<std::mutex> locker(_mu);
		_cond.wait(locker, [this]() { return !_queue.empty(); }); // In case spurious wake
		T output = _queue.front();
		_queue.pop();
		return output;
	}
private:
	std::queue<T> _queue;
	std::mutex _mu;
	std::condition_variable _cond;
};


cv::Mat render_event(std::vector<Event>& events, int width, int height);

void event_generation_wrapper(
	EventSimulator& esim,
	SafeQueue<cv::Mat>& og_image_queue,
	SafeQueue<std::vector<Event>>& event_queue,
	double fps,
	int frame_count);
void render_event_wrapper(
	SafeQueue<std::vector<Event>>& event_queue, 
	SafeQueue<cv::Mat>& rendered_img_queue,
	int width, int height, int frame_count);
void write_video_wrapper(SafeQueue<cv::Mat>& rendered_img_queue,const int fourcc, const int fps, const cv::Size size, const int frame_count);
void show_vid(std::queue<cv::Mat>& rendered_img_queue);


int main(int argc, char* argv[]){
		std::string file_path = argv[1];

		float Cp=0.1f, Cn=0.1f;
		float refractory_period=1e-4f;
		float log_eps=1e-3f;
		bool use_log=true;
		uint32_t width, height;

		EventSimulator esim(Cp, Cn, refractory_period, log_eps, use_log);

		SafeQueue<cv::Mat> og_image_queue;
		SafeQueue<std::vector<Event>> event_queue;
		SafeQueue<cv::Mat> rendered_img_queue;

		cv::VideoCapture vcap(file_path);
		{
			cv::Mat test_image;
			vcap >> test_image;
			if (test_image.empty())
				return EXIT_FAILURE;
		}
		float fps = vcap.get(cv::CAP_PROP_FPS);
		int frame_count = vcap.get(cv::CAP_PROP_FRAME_COUNT) - 1 ;
		width = vcap.get(cv::CAP_PROP_FRAME_WIDTH);
		height = vcap.get(cv::CAP_PROP_FRAME_HEIGHT);
		int codec = 0x00000021;

		std::thread event_generation_thread(event_generation_wrapper, std::ref(esim), std::ref(og_image_queue), std::ref(event_queue), fps, frame_count);
		std::thread render_thread(render_event_wrapper, std::ref(event_queue), std::ref(rendered_img_queue), width, height, frame_count);
		std::thread write_thread(write_video_wrapper, std::ref(rendered_img_queue), codec, fps, cv::Size(width, height), frame_count);
		// std::thread show_thread(show_vid, std::ref(rendered_img_queue));

		for (int frame_number = 0; frame_number < frame_count; ++frame_number) {
			int new_width, new_height, col_from, col_to;
			cv::Mat curr_image(cv::Size(width, height), CV_8UC3), gray_image(cv::Size(width, height), CV_8UC1);
			vcap >> curr_image;
			cv::cvtColor(curr_image, gray_image, cv::COLOR_BGR2GRAY);
			gray_image.convertTo(gray_image, CV_32F, 1.0 / 255);
			og_image_queue.push(gray_image);
		}


		event_generation_thread.join();
		render_thread.join();
		write_thread.join();
		// show_thread.join();

		return EXIT_SUCCESS;
}

void event_generation_wrapper(
	EventSimulator& esim,
	SafeQueue<cv::Mat>& og_image_queue,
	SafeQueue< std::vector<Event> >& event_queue,
	double fps, int frame_count
	){
	for (int frame_number=0; frame_number < frame_count; ++frame_number){
		double time_msec = 1e3 * frame_number / fps;
		cv::Mat curr_image = og_image_queue.front(), log_img;
		if (esim.IsUseLog())
			cv::log(curr_image + esim.getLogEps(), log_img);
		std::vector<Event> events;
		esim.imageCallback(curr_image, time_msec, events); 
		event_queue.push(events);
	}
		
}

void render_event_wrapper(
	SafeQueue< std::vector<Event> >& event_queue, 
	SafeQueue<cv::Mat>& rendered_img_queue,
	int width, int height, int frame_count){
	for (int frame_number = 0; frame_number < frame_count; ++frame_number) {
		std::vector<Event> events = event_queue.front();
		if (events.size()) {
			cv::Mat rendered_image = render_event(events, width, height);
			rendered_img_queue.push(rendered_image);
		}
	}			
}

void write_video_wrapper(SafeQueue<cv::Mat>& rendered_img_queue, const int fourcc, const int fps, const cv::Size size, const int frame_count){
	cv::VideoWriter writer("output.mp4", fourcc, fps, size);
	std::filesystem::create_directory("output");
	for (int frame_number=0; frame_number < (frame_count - 1); ++frame_number) {
		cv::Mat curr_image = rendered_img_queue.front();
		writer.write(curr_image);
		std::cout << "wrote " << frame_number << "/" << (frame_count-1) << " of image \r";
	}

}


cv::Mat render_event(std::vector<Event>& events, int width, int height){
	cv::Mat pos_event(height, width, CV_8UC1, cv::Scalar(0)), 
		neg_event(height, width, CV_8UC1, cv::Scalar(0)),
		mask(height, width, CV_8UC1, cv::Scalar(0));

	// Drawing The image
	uchar render_intensity=100;
	for(Event& event: events){
		if (event.polarity_ == 1.0){ // Positive Event has +1 polarity
			pos_event.at<uchar>(cv::Point(event.x_, event.y_)) = render_intensity;
		}else if (event.polarity_ == -1.0){
			neg_event.at<uchar>(cv::Point(event.x_, event.y_)) = render_intensity;
		}
	}


	std::vector<cv::Mat> channels;
	channels.push_back(mask); // B
	channels.push_back(neg_event); // G
	channels.push_back(pos_event); // R

	cv::Mat final_img;

	cv::merge(channels, final_img);
	return final_img;
}

void show_vid(std::queue<cv::Mat>& rendered_img_queue){

	while(true){
		if(!rendered_img_queue.empty()){
			cv::Mat curr_img = rendered_img_queue.front();
			rendered_img_queue.pop();

			cv::imshow("Current_image", curr_img);
		}
	}
}



