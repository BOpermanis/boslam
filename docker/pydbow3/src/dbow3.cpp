// Author: Andrius Mikonis (andrius.mikonis@gmail.com)
// License: BSD
// Last modified: Feb 12, 2019

// Wrapper for most external modules
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <exception>

// Opencv includes
#include <opencv2/opencv.hpp>

// np_opencv_converter
#include "np_opencv_converter.hpp"

// DBoW3
#include "DBoW3.h"

namespace py = boost::python;

cv::Mat test_np_mat(const cv::Mat& in) {
	std::cerr << "in: " << in << std::endl;
	std::cerr << "sz: " << in.size() << std::endl;
	return in.clone();
}

cv::Mat test_with_args(const cv::Mat_<float>& in, const int& var1 = 1,
	const double& var2 = 10.0, const std::string& name = std::string("test_name")) {
	std::cerr << "in: " << in << std::endl;
	std::cerr << "sz: " << in.size() << std::endl;
	std::cerr << "Returning transpose" << std::endl;
	return in.t();
}

class GenericWrapper {
public:
	GenericWrapper(const int& _var_int = 1, const float& _var_float = 1.f,
		const double& _var_double = 1.f, const std::string& _var_string = std::string("test_string"))
		: var_int(_var_int), var_float(_var_float), var_double(_var_double), var_string(_var_string)
	{

	}

	cv::Mat process(const cv::Mat& in) {
		std::cerr << "in: " << in << std::endl;
		std::cerr << "sz: " << in.size() << std::endl;
		std::cerr << "Returning transpose" << std::endl;
		return in.t();
	}

private:
	int var_int;
	float var_float;
	double var_double;
	std::string var_string;
};

class Vocabulary
{
public:
	Vocabulary(int k = 10, int L = 5, DBoW3::WeightingType weighting = DBoW3::TF_IDF, DBoW3::ScoringType scoring = DBoW3::L1_NORM, const std::string& path = std::string()) {
		vocabulary = new DBoW3::Vocabulary(k, L, weighting, scoring);
		if (!path.empty())
			load(path);
	}
	~Vocabulary() {
		delete vocabulary;
	}
	
	void create(const  std::vector<cv::Mat>   &training_features) {
		vocabulary->create(training_features);
	}
	
	void clear() {
		vocabulary->clear();
	}

	void load(const std::string& path) {
		vocabulary->load(path);
	}

	void save(const std::string& path, bool binary_compressed = true) {
		vocabulary->save(path, binary_compressed);
	}

	DBoW3::BowVector transform(const  std::vector<cv::Mat> & features) {
		DBoW3::BowVector word;
		vocabulary->transform(features, word);
		return word;
	}

	DBoW3::WordId feat_id(const cv::Mat& feature) {
		return vocabulary->transform(feature);
	}

	std::vector<unsigned int> feats2words(const std::vector<cv::Mat> &features) {
		DBoW3::BowVector bow;
		vocabulary->transform(features, bow);
		std::vector<unsigned int> words;
		for(auto it=bow.begin(); it!=bow.end(); it++){
		    words.push_back(it->first);
		}
		return words;
	}

	DBoW3::WordId id_weight(const DBoW3::WordId& i) {
		return vocabulary->getWordWeight(i);
	}

    int distance(const cv::Mat& f1, const cv::Mat& f2) {
        return vocabulary->hamming_distance(f1.t(), f2.t());
    }

	double score(const  DBoW3::BowVector &A, const DBoW3::BowVector &B) {
		return vocabulary->score(A, B);
	}

	DBoW3::Vocabulary * vocabulary;
};

class Database
{
public:
	Database(const std::string& path = std::string()) {
		if (path.empty())
			database = new DBoW3::Database();
		else
			database = new DBoW3::Database(path);
	}
	~Database() {
		delete database;
	}

	void setVocabulary(const Vocabulary& vocabulary, bool use_di, int di_levels=0) {
		database->setVocabulary(*vocabulary.vocabulary, use_di, di_levels);
	}

	unsigned int add(const  cv::Mat & features) {
		return database->add(features, NULL, NULL);
	}

    void erase(const int i){
        DBoW3::WordId wid = (DBoW3::WordId) i;
        database->erase(wid);
    }

	std::vector<DBoW3::Result> query(const  cv::Mat &features, const std::vector<int> target_inds, int max_results = 1, int max_id = -1) {
		DBoW3::QueryResults results;
//		std::cout << set_target.empty() << std::endl;
		database->query(features, results, target_inds, max_results, max_id);
		return results;
	}

    cv::Mat compare_bows(const int i, const int j) {
        cv::Mat out;
        unsigned int cnt = 0;
        float score = 0;
        database->compareBowsL1(i, j, cnt, score);
        out.push_back(float(cnt));
        out.push_back(score);
        return out;
    }

    std::vector<int> commonWords(const cv::Mat &features, const DBoW3::WordId i) {
        std::vector<DBoW3::WordId> inds;
        std::vector<int> inds1;
        database->commonWords(features, i, inds);
        for(auto it = inds.begin(); it!=inds.end(); it++) {
            inds1.push_back((int)*it);
        }
//        inds1 = (std::vector<DBoW3::WordId>) inds;
//        inds.push_back(12);
        return inds1;
    }

	void save(const std::string &filename) const {
		database->save(filename);
	}

	void load(const std::string &filename) {
		database->load(filename);
	}

	void loadVocabulary(const std::string &filename, bool use_di, int di_levels=0) {
		DBoW3::Vocabulary voc;
		voc.load(filename);
		database->setVocabulary(voc, use_di, di_levels);
	}


private:
	DBoW3::Database * database;
};

namespace fs {
	namespace python {

		BOOST_PYTHON_MODULE(pyDBoW3)
		{
			// Main types export
			fs::python::init_and_export_converters();
			py::scope scope = py::scope();

			py::enum_<DBoW3::WeightingType>("WeightingType")
				.value("TF_IDF", DBoW3::TF_IDF)
				.value("TF", DBoW3::TF)
				.value("IDF", DBoW3::IDF)
				.value("BINARY", DBoW3::BINARY);

			py::enum_<DBoW3::ScoringType>("ScoringType")
				.value("L1_NORM", DBoW3::L1_NORM)
				.value("L2_NORM", DBoW3::L2_NORM)
				.value("CHI_SQUARE", DBoW3::CHI_SQUARE)
				.value("KL", DBoW3::KL)
				.value("BHATTACHARYYA", DBoW3::BHATTACHARYYA)
				.value("DOT_PRODUCT", DBoW3::DOT_PRODUCT);

			// Class
			py::class_<Vocabulary>("Vocabulary")
				.def(py::init<py::optional<int, int, DBoW3::WeightingType, DBoW3::ScoringType, std::string> >(
				(py::arg("k") = 10, py::arg("L") = 5, py::arg("weighting") = DBoW3::TF_IDF, py::arg("scoring") = DBoW3::L1_NORM,
					py::arg("path") = std::string())))
				.def("load", &Vocabulary::load)
				.def("save", &Vocabulary::save)
				.def("create", &Vocabulary::create)
				.def("transform", &Vocabulary::transform, py::return_value_policy<py::return_by_value>())
				.def("feat_id", &Vocabulary::feat_id, py::return_value_policy<py::return_by_value>())
				.def("id_weight", &Vocabulary::id_weight, py::return_value_policy<py::return_by_value>())
				.def("feats2words", &Vocabulary::feats2words, py::return_value_policy<py::return_by_value>())
				.def("distance", &Vocabulary::distance, py::return_value_policy<py::return_by_value>())
				.def("clear", &Vocabulary::clear);

			py::class_<Database>("Database")
				.def(py::init<py::optional<std::string> >(py::arg("path") = std::string()))
				.def("setVocabulary", &Database::setVocabulary)
				.def("save", &Database::save)
				.def("load", &Database::load)
				.def("loadVocabulary", &Database::loadVocabulary)
				.def("add", &Database::add)
				.def("compare_bows", &Database::compare_bows, py::return_value_policy<py::return_by_value>())
				.def("commonWords", &Database::commonWords, py::return_value_policy<py::return_by_value>())
				.def("erase", &Database::erase)
				.def("query", &Database::query, py::return_value_policy<py::return_by_value>());

			py::class_<DBoW3::Result>("Result")
				.def_readonly("Id", &DBoW3::Result::Id)
				.def_readonly("Score", &DBoW3::Result::Score)
				.def_readonly("nWords", &DBoW3::Result::nWords)
				.def_readonly("bhatScore", &DBoW3::Result::bhatScore)
				.def_readonly("chiScore", &DBoW3::Result::chiScore)
				.def_readonly("sumCommonVi", &DBoW3::Result::sumCommonVi)
				.def_readonly("sumCommonWi", &DBoW3::Result::sumCommonWi)
				.def_readonly("minScoreInSelection", &DBoW3::Result::minScoreInSelection)
				.def_readonly("expectedChiScore", &DBoW3::Result::expectedChiScore);

		}

	} // namespace fs
} // namespace python
