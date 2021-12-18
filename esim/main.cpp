#include <iostream>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <iterator>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "esim"

cv::Mat render_event(std::vector<Event>& events, int width, int height);

void event_generation_wrapper(
	EventSimulator& esim,
	std::queue<cv::Mat>& og_image_queue,
	std::queue<std::vector<Event>>& event_queue,
	double fps,
	int frame_count);
void render_event_wrapper(
	std::queue<std::vector<Event>>& event_queue, 
	std::queue<cv::Mat>& rendered_img_queue,
	int width, int height, int frame_count);
void write_video_wrapper(std::queue<cv::Mat>& rendered_img_queue,const int fourcc, const int fps, const cv::Size size, const int frame_count);
void show_vid(std::queue<cv::Mat>& rendered_img_queue);


int main(int argc, char* argv[]){
		std::string file_path = argv[0];

		float Cp=0.1, Cn=0.1;
		float refractory_period=1e-4;
		float log_eps=1e-3;
		bool use_log=true;
		uint32_t width, height;

		EventSimulator esim(Cp, Cn, refractory_period, log_eps, use_log);

		std::queue<cv::Mat> og_image_queue;
		std::queue<std::vector<Event>> event_queue;
		std::queue<cv::Mat> rendered_img_queue;
		cv::VideoCapture vcap(file_path);
		{
			cv::Mat test_image; 
			vcap >> test_image;
		}
		float fps = vcap.get(cv::CAP_PROP_FPS);
		auto frame_count = vcap.get(cv::CAP_PROP_FRAME_COUNT);
		width = vcap.get(cv::CAP_PROP_FRAME_WIDTH);
		height = vcap.get(cv::CAP_PROP_FRAME_WIDTH);
		int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');

		std::thread event_generation_thread(event_generation_wrapper, std::ref(esim), std::ref(og_image_queue), std::ref(event_queue), fps, frame_count);
		std::thread render_thread(render_event_wrapper, std::ref(event_queue), std::ref(rendered_img_queue), width, height, frame_count);
		std::thread write_thread(write_video_wrapper, std::ref(rendered_img_queue), codec, fps, cv::Size(width, height), frame_count);
		// std::thread show_thread(show_vid, std::ref(rendered_img_queue));

		for (int frame_number = 0; frame_number < frame_count; ++frame_number) {
			int new_width, new_height, col_from, col_to;
			cv::Mat curr_image;
			vcap >> curr_image;
			og_image_queue.push(curr_image);

		}


		event_generation_thread.join();
		render_thread.join();
		write_thread.join();
		// show_thread.join();

		return EXIT_SUCCESS;
}

void event_generation_wrapper(
	EventSimulator& esim,
	std::queue<cv::Mat>& og_image_queue,
	std::queue<std::vector<Event>>& event_queue,
	double fps, int frame_count
	){
	while (og_image_queue.empty())
		continue; // Wait until not empty


	for (int frame_number=0; frame_number < frame_count; ++frame_number){
		double time_msec = 1e3 * frame_number / fps;
		cv::Mat curr_image = og_image_queue.front();
		og_image_queue.pop();
		std::vector<Event> events;
		esim.imageCallback(curr_image, time_msec, events); 
		event_queue.push(events);
	}
		
}

void render_event_wrapper(
	std::queue<std::vector<Event>>& event_queue, 
	std::queue<cv::Mat>& rendered_img_queue,
	int width, int height, int frame_count){
	for (int frame_number = 0; frame_number < frame_count; ++frame_number) {
		if(!event_queue.empty()){

			std::vector<Event>& events = event_queue.front();
			event_queue.pop();
			cv::Mat rendered_image = render_event(events, width, height);

			rendered_img_queue.push(rendered_image);
		}else{
			continue;
		}

	}
				
}

void write_video_wrapper(std::queue<cv::Mat>& rendered_img_queue, const int fourcc, const int fps, const cv::Size size, const int frame_count){
	cv::VideoWriter writer("output", fourcc, fps, size);
	while (rendered_img_queue.empty())
		continue;
	for (int frame_number; frame_number < frame_count; ++frame_number) {
		cv::Mat curr_image = rendered_img_queue.front();
		rendered_img_queue.pop();
		writer.write(curr_image);
	}

}


cv::Mat render_event(std::vector<Event>& events, int width, int height){
	cv::Mat pos_event(height, width, CV_8UC1), neg_event(height, width, CV_8UC1);
	cv::Mat mask(height, width, CV_8UC1, cv::Scalar(0));

	// Drawing The image
	std::vector<Event>::iterator event;
	for(event = events.begin(); event != events.end(); ++event){
		if (event->polarity_ == 1.0){ // Positive Event has +1 polarity
			pos_event.at<uchar>(cv::Point(event->x_, event->y_)) = 50;
		}else if (event->polarity_ == -1.0){
			neg_event.at<uchar>(cv::Point(event->x_, event->y_)) = 50;
		}
	}


	std::vector<cv::Mat> channels;
	channels.push_back(pos_event);
	channels.push_back(mask);
	channels.push_back(neg_event);

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



