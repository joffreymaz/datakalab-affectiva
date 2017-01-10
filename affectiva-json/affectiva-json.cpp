#include <iostream>
#include <typeinfo>
#include <memory>
#include <chrono>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/timer/timer.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>


#include "Frame.h"
#include "Face.h"
#include "FrameDetector.h"
#include "AffdexException.h"

#include "AFaceListener.hpp"
#include "PlottingImageListener.hpp"
#include "StatusListener.hpp"

using namespace std;
using namespace affdex;
namespace pt = boost::property_tree;

int main(int argsc, char ** argsv)
{
    namespace po = boost::program_options; // abbreviate namespace
    std::string fileOutput = "experience_data/experience" +
      boost::posix_time::to_iso_string(boost::posix_time::second_clock::local_time()) +
					  ".json";
    shared_ptr<FrameDetector> frameDetector;
    try{
        const std::vector<int> DEFAULT_RESOLUTION{ 1920, 1080 };

	std::string dataFolder = "/opt/affdex-sdk/data";

        std::vector<int> resolution;
        int process_framerate = 20;
        int camera_framerate = 30;
	int cameraResHeight = 720;
	int cameraResWidth = 480;
        int buffer_length = 30;
        int camera_id = 0;
        unsigned int nFaces = 8;
        bool draw_display = true;
        int faceDetectorMode = (int)FaceDetectorMode::LARGE_FACES;

        float last_timestamp = -1.0f;
        float capture_fps = -1.0f;

        const int precision = 2;
        std::cerr.precision(precision);
        std::cout.precision(precision);

        std::ofstream csvFileStream;
        std::cerr << "Initializing Affdex FrameDetector" << endl;
        shared_ptr<FaceListener> faceListenPtr(new AFaceListener());
        shared_ptr<PlottingImageListener> listenPtr(new PlottingImageListener(csvFileStream, draw_display));    // Instanciate the ImageListener class
        shared_ptr<StatusListener> videoListenPtr(new StatusListener());
	frameDetector = make_shared<FrameDetector>(buffer_length, process_framerate, nFaces, (affdex::FaceDetectorMode)faceDetectorMode);
        //Initialize detectors
        frameDetector->setDetectAllEmotions(true);
        frameDetector->setDetectAllExpressions(true);
        frameDetector->setDetectAllEmojis(true);
        frameDetector->setDetectAllAppearances(true);
        frameDetector->setClassifierPath(dataFolder);
        frameDetector->setImageListener(listenPtr.get());
        frameDetector->setFaceListener(faceListenPtr.get());
        frameDetector->setProcessStatusListener(videoListenPtr.get());

        cv::VideoCapture webcam(camera_id);    //Connect to the first webcam
        webcam.set(CV_CAP_PROP_FPS, camera_framerate);    //Set webcam framerate.
        webcam.set(CV_CAP_PROP_FRAME_WIDTH, cameraResWidth);
        webcam.set(CV_CAP_PROP_FRAME_HEIGHT, cameraResHeight);
        auto start_time = std::chrono::system_clock::now();
        if (!webcam.isOpened())
        {
            std::cerr << "Error opening webcam!" << std::endl;
            return 1;
        }
        //Start the frame detector thread.
        frameDetector->start();
        do{
            cv::Mat img;
	    // Display camera feedback
            if (!webcam.read(img))    //Capture an image from the camera
            {
                std::cerr << "Failed to read frame from webcam! " << std::endl;
                break;
            }
            //Calculate the Image timestamp and the capture frame rate;
            const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);
            const double seconds = milliseconds.count() / 1000.f;
            // Create a frame
            Frame f(img.size().width, img.size().height, img.data, Frame::COLOR_FORMAT::BGR, seconds);
            capture_fps = 1.0f / (seconds - last_timestamp);
            last_timestamp = seconds;
            frameDetector->process(f);  //Pass the frame to detector
            // For each frame processed
            if (listenPtr->getDataSize() > 0)
            {
                std::pair<Frame, std::map<FaceId, Face> > dataPoint = listenPtr->getData();
                Frame frame = dataPoint.first;
                std::map<FaceId, Face> faces = dataPoint.second;
                // Draw metrics to the GUI
		if (draw_display)
		  {
                    listenPtr->draw(faces, frame);
		  }
		// Output in json
		pt::ptree root;
		pt::ptree person;
		pt::ptree emotions;
		pt::ptree expressions;
		pt::ptree headOrientation;
		pt::ptree emojis;
		root.put("Timestamp_experience", frame.getTimestamp());
		root.put("Timestamp_local", boost::posix_time::to_simple_string(boost::posix_time::second_clock::local_time()));
		for (int i = 0; i < faces.size(); i++)
		  {
		    emotions.put("Joy", faces[i].emotions.joy);
		    emotions.put("Fear", faces[i].emotions.fear);
		    emotions.put("Disgust", faces[i].emotions.disgust);
		    emotions.put("Sadness", faces[i].emotions.sadness);
		    emotions.put("Anger", faces[i].emotions.anger);
		    emotions.put("Surprise", faces[i].emotions.surprise);
		    emotions.put("Contempt", faces[i].emotions.contempt);
		    emotions.put("Valence", faces[i].emotions.valence);
		    emotions.put("Engagement", faces[i].emotions.engagement);
		    emojis.put("Relaxed", faces[i].emojis.relaxed);
		    emojis.put("Smiley", faces[i].emojis.smiley);
		    emojis.put("Laughing", faces[i].emojis.laughing);
		    emojis.put("Kissing", faces[i].emojis.kissing);
		    emojis.put("Disappointed", faces[i].emojis.disappointed);
		    emojis.put("Rage", faces[i].emojis.rage);
		    emojis.put("Smirk", faces[i].emojis.smirk);
		    emojis.put("Wink", faces[i].emojis.wink);
		    emojis.put("stuckOutTongueWinkingEye", faces[i].emojis.stuckOutTongueWinkingEye);
		    emojis.put("stuckOutTongue", faces[i].emojis.stuckOutTongue);
		    emojis.put("Flushed", faces[i].emojis.flushed);
		    emojis.put("Scream", faces[i].emojis.scream);
		    expressions.put("Smile", faces[i].expressions.smile);
		    expressions.put("InnerBrowRaise", faces[i].expressions.innerBrowRaise);
		    expressions.put("BrowRaise", faces[i].expressions.browRaise);
		    expressions.put("BrowFurrow", faces[i].expressions.browFurrow);
		    expressions.put("NoseWrinkle", faces[i].expressions.noseWrinkle);
		    expressions.put("UpperLipRaise", faces[i].expressions.upperLipRaise);
		    expressions.put("LipCornerDepressor", faces[i].expressions.lipCornerDepressor);
		    expressions.put("ChinRaise", faces[i].expressions.chinRaise);
		    expressions.put("LipPucker", faces[i].expressions.lipPucker);
		    expressions.put("LipPress", faces[i].expressions.lipPress);
		    expressions.put("LipSuck", faces[i].expressions.lipSuck);
		    expressions.put("MouthOpen", faces[i].expressions.mouthOpen);
		    expressions.put("Smirk", faces[i].expressions.smirk);
		    expressions.put("EyeClosure", faces[i].expressions.eyeClosure);
		    expressions.put("Attention", faces[i].expressions.attention);
		    expressions.put("EyeWiden", faces[i].expressions.eyeWiden);
		    expressions.put("CheekRaise", faces[i].expressions.cheekRaise);
		    expressions.put("LidTighten", faces[i].expressions.lidTighten);
		    expressions.put("Dimpler", faces[i].expressions.dimpler);
		    expressions.put("LipStretch", faces[i].expressions.lipStretch);
		    expressions.put("JawDrop", faces[i].expressions.jawDrop);
		    //  Relaxed = 9786, Smiley = 128515, Laughing = 128518, Kissing = 128535,
		    // Disappointed = 128542, Rage = 128545, Smirk = 128527, Wink = 128521,
		    //  StuckOutTongueWinkingEye = 128540, StuckOutTongue = 128539, Flushed = 128563, Scream = 128561,
		    //Unknown = 128528
		    //emojis.put("DominantEmoji", faces[0].emojis.dominantEmoji);
		    //Age Values {AGE_UNKNOWN, AGE_UNDER_18, AGE_18_24, AGE_25_34, AGE_35_44, AGE_45_54, AGE_55_64, AGE_65_PLUS}
		    headOrientation.put("Pitch", faces[i].measurements.orientation.pitch);
		    headOrientation.put("Yaw", faces[i].measurements.orientation.yaw);
		    headOrientation.put("Roll", faces[i].measurements.orientation.roll);
		    person.put("Id", i);
		    person.put("Age", faces[i].appearance.age);
		    person.put("InterocularDistance", faces[i].measurements.interocularDistance);
		    // Unknwon, Male or Female
		    //person.put("Gender", faces[i].appearance.gender);
		    //std::cout << faces[i] << std::endl;
		    //Ethnicity : UNKNOWN, CAUCASIAN, BLACK_AFRICAN, SOUTH_ASIAN, EAST_ASIAN, HISPANIC
		    person.put("Ethnicity", faces[i].appearance.ethnicity);
		    // Glasses : No, yes
		    // person.put("Glasses", faces[i].appearance.glasses;
		    person.add_child("HeadOrientation", headOrientation);
		    person.add_child("Emotions", emotions);
		    person.add_child("Emojis", emojis);
		    person.add_child("Expressions", expressions);
		    root.add_child("Person", person);
		  }
		std::ofstream outFile;
		outFile.open(fileOutput, ios::out | ios::app);
		pt::write_json(outFile, root);
		// Debug
		std::cerr << " cfps: " << listenPtr->getCaptureFrameRate()
                    << " pfps: " << listenPtr->getProcessingFrameRate()
                    << " faces: " << faces.size() << endl;
            }
        }
        while (videoListenPtr->isRunning());//(cv::waitKey(20) != -1);
	frameDetector->stop();    //Stop frame detector thread
    }
    catch (AffdexException ex)
      {
        std::cerr << "Encountered an AffdexException " << ex.what();
        return 1;
      }
    catch (std::runtime_error err)
      {
        std::cerr << "Encountered a runtime error " << err.what();
        return 1;
      }
    catch (std::exception ex)
      {
        std::cerr << "Encountered an exception " << ex.what();
        return 1;
      }
    catch (...)
      {
        std::cerr << "Encountered an unhandled exception ";
        return 1;
      }

    return 0;
}
