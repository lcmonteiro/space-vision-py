/**
 * ------------------------------------------------------------------------------------------------
 * File:   vision_detection_app.h
 * Author: Luis Monteiro
 *
 * Created on oct 8, 2019, 22:00 PM
 * ------------------------------------------------------------------------------------------------
 **
 * std
 */
#include <fstream>
#include <iostream>
/**
 * remote
 */
#include <args.hxx>
/**
 * local
 */
#include "vision_detection.hpp"
/**
 * ----------------------------------------------------------------------------
 * main
 * ----------------------------------------------------------------------------
 */
int main(int argc, char* argv[]) {
	/**
	 * ----------------------------------------------------
	 * parse arguments configuration
	 * ----------------------------------------------------
	 */
	args::ArgumentParser parser("vision detection");
    args::HelpFlag help(parser, 
		"HELP", 
		"Show this help menu.", 
		{'h', "help"}
	);
    args::ValueFlag<std::string> path(parser, 
		"PATH", 
		"configuration path", 
		{'c', "config"}, 
		"vision_detection_app.yaml"
	);
	/**
	 * ----------------------------------------------------
	 * parse arguments
	 * ----------------------------------------------------
	 */
    try {
        parser.ParseCLI(argc, argv);
    } catch (args::Help) {
        std::cout << parser;
        return 0;
    } catch (args::ParseError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    } catch (args::ValidationError e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
	/**
	 * ----------------------------------------------------
	 * init and load filters 
	 * ----------------------------------------------------
	 */				
	VisionDetection vd(std::ifstream(args::get(path)));
	/**
	 * ----------------------------------------------------
	 * configure filter
	 * ----------------------------------------------------
	 */
	vd.set_filter_all();
	/**
	 * ----------------------------------------------------
	 * run detection
	 * ----------------------------------------------------
	 */
	return vd.serve([](auto id, auto result) {
		std::cout << "result::id=" << id << std::endl;
	});
}
/**
 * ------------------------------------------------------------------------------------------------
 * End
 * ------------------------------------------------------------------------------------------------
 */
