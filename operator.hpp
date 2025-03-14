/*

only e4m3 format supported

*/

#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include <tuple>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>
#include <iostream>

class operands {
private:
	static constexpr int _num_exp = 4;
	static constexpr int _num_man = 3;

	static std::tuple<int, float, float, float> efm_decoder(uint8_t fp8_bin);
	static float fp8_dec_value(uint8_t fp8_bin);
	static float lmul_single(float fp32_dec_x, float fp32_dec_y);
	static std::pair<uint8_t, float> fp32_downcast(float fp32_dec);
public:
	static float fp32_ele_downcast(float fp32_dec);
	static std::vector<std::vector<float>> fp32_mat_downcast(std::vector<std::vector<float>> fp32_mat);
	static std::vector<std::vector<float>> lmul_matmul(std::vector<std::vector<float>> fp32_mat_x, std::vector<std::vector<float>> fp32_mat_y);
};

#endif