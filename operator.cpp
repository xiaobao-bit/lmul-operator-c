#include "operator.hpp"

/*
function to transfer fp8(bin) into 4 parts

    input:  fp8 in bin
    output: tuple <sign, E, f, M>
*/
std::tuple<int, float, float, float> operands::efm_decoder(uint8_t fp8_bin) {
	int num_exp = operands::_num_exp;
	int num_man = operands::_num_man;

	uint8_t exp_mask = 0x78;
	uint8_t man_mask = 0x07;

	int sign = (fp8_bin & 0x80) >> 7;
	int exp = (fp8_bin & exp_mask) >> num_man;
	int man = fp8_bin & man_mask;

	int bias = (1 << (num_exp - 1)) - 1;
	float E = (exp == 0) ? 0 : (exp - bias);
	float f = man / static_cast<float>(1 << num_man);
	float M = (exp == 0) ? 0 : (1.0f + f);

	return { sign, E, f, M };
}

/*
function to decode fp8(bin) into fp8(dec)
    
    input:  fp8 in bin
    output: fp8 in dec
*/
float operands::fp8_dec_value(uint8_t fp8_bin) {
	auto tpl_fp8_dec_val = operands::efm_decoder(fp8_bin);
	int sign = std::get<0>(tpl_fp8_dec_val);
	float E = std::get<1>(tpl_fp8_dec_val);
	float M = std::get<3>(tpl_fp8_dec_val);
	return (sign ? -1.0f : 1.0f) * std::powf(2.0f, E) * M;
}

/*
function to downcast fp32(dec) into fp8

    input:  fp32 in dec
    output: pair <fp8 in bin, fp8 in dec>
*/
std::pair<uint8_t, float> operands::fp32_downcast(float fp32_dec) {
    const int source_m_nbits = 23;  // FP32 mantissa bits
    const int source_e_nbits = 8;   // FP32 exponent bits
    const int target_e_nbits = operands::_num_exp;
    const int target_m_nbits = operands::_num_man;

    if (target_e_nbits + target_m_nbits + 1 > 8) {
        throw std::invalid_argument("Mantissa is too large for an 8-bit float");
    }

    // IEEE 754 bit representation
    union {
        float f;
        uint32_t i;
    } u;
    u.f = fp32_dec;

    // Extract sign, exponent, and mantissa
    uint8_t sign = (u.i >> 31) & 0x1;
    uint32_t fp32_exp = (u.i >> source_m_nbits) & 0xFF;
    uint32_t fp32_mantissa = u.i & ((1u << source_m_nbits) - 1);

    // Compute FP8 bias
    int fp32_bias = 127;
    int fp8_bias = (1 << (target_e_nbits - 1)) - 1;

    // Compute new exponent
    int fp8_exp = fp32_exp - fp32_bias + fp8_bias;
    bool is_subnormal = false;

    // Handle subnormal numbers
    if (fp8_exp < 1) {
        is_subnormal = true;
        fp8_exp = 0;
        int shift = 1 - (fp32_exp - fp32_bias);
        fp32_mantissa = (0x800000 | fp32_mantissa) >> shift;
    }
    // Handle exponent overflow (set to infinity)
    else if (fp8_exp >= (1 << target_e_nbits)) {
        fp8_exp = (1 << target_e_nbits) - 1;
        fp32_mantissa = (1 << target_m_nbits) - 1;
    }

    fp8_exp = std::max(1, std::min(fp8_exp, (1 << target_e_nbits) - 1));

    // Truncate mantissa
    int mantissa_shift = source_m_nbits - target_m_nbits;
    uint8_t truncated_mantissa = fp32_mantissa >> mantissa_shift;

    // Compute probability for stochastic rounding
    uint32_t remainder = fp32_mantissa & ((1 << mantissa_shift) - 1);
    float probability = static_cast<float>(remainder) / (1 << mantissa_shift);

    // Random rounding decision
    static std::mt19937 gen{ std::random_device{}() };
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (dist(gen) < probability && truncated_mantissa < (1 << target_m_nbits) - 1) {
        truncated_mantissa += 1;
    }
    if (truncated_mantissa >= (1 << target_m_nbits)) {
        truncated_mantissa = 0;
        fp8_exp += 1;
        if (fp8_exp >= (1 << target_e_nbits)) {
            fp8_exp = (1 << target_e_nbits) - 1;
            truncated_mantissa = (1 << target_m_nbits) - 1;
        }
    }
    // Compose final FP8 representation
    uint8_t fp8_bin = (sign << 7) | (fp8_exp << target_m_nbits) | truncated_mantissa;
    float fp8_dec = operands::fp8_dec_value(fp8_bin);

    auto output = std::make_pair (fp8_bin, fp8_dec);
    return output;
}

/*
function of lmul with single element

    input:  2 fp32 in dec
    output: lmul result in dec
*/
float operands::lmul_single(float fp32_dec_x, float fp32_dec_y) {
    int num_exp = operands::_num_exp;
    int num_man = operands::_num_man;

    auto fp8_x = operands::fp32_downcast(fp32_dec_x);
    auto fp8_y = operands::fp32_downcast(fp32_dec_y);

    std::tuple<int, float, float, float> res_x = operands::efm_decoder(fp8_x.first);
    std::tuple<int, float, float, float> res_y = operands::efm_decoder(fp8_y.first);

    int lm;
    if (num_man <= 3) lm = num_man;
    else if (num_man == 4) lm = 3;
    else lm = 4;

    int sign = (std::get<0>(res_x) == std::get<0>(res_y)) ? 1 : -1;

    float Ex = std::get<1>(res_x);
    float Ey = std::get<1>(res_y);
    float fx = std::get<2>(res_x);
    float fy = std::get<2>(res_y);

    // #pragma omp parallel for collapse(2) num_threads(8)
    float res = sign * (1.0f + fx + fy + std::powf(2.0f, -lm)) * std::powf(2.0f, Ex + Ey);
    // float res = sign * (1.0f + fx) * (1.0f + fy) * std::powf(2.0f, Ex + Ey);
    return res;
}

/*
function to transfer a fp32 element into fp8 element

    input:  fp32 element in dec
    output: fp8 element in dec
*/
float operands::fp32_ele_downcast(float fp32_dec) {
    return operands::fp32_downcast(fp32_dec).second;
}

/*
function to transfer a fp32 mat into fp8 mat

    input:  fp32 mat in dec
    output: fp8 mat in dec
*/
std::vector<std::vector<float>> operands::fp32_mat_downcast(std::vector<std::vector<float>> fp32_mat) {
    if (fp32_mat.empty()) {
        throw std::invalid_argument("Empty input matrix");
    }

    std::vector<std::vector<float>> fp8_mat(fp32_mat.size(), std::vector<float>(fp32_mat[0].size()));

    #pragma omp parallel for collapse(2) num_threads(8)
    for (size_t i = 0; i < fp32_mat.size(); i++) {
        for (size_t j = 0; j < fp32_mat[i].size(); j++) {
            fp8_mat[i][j] = operands::fp32_downcast(fp32_mat[i][j]).second;
        }
    }
    return fp8_mat;
}

/*]
function to do the lmul mat. multiplication

    input:  2 fp32 mat in dec
    output: result mat of lmul in dec
*/
std::vector<std::vector<float>> operands::lmul_matmul(std::vector<std::vector<float>> fp32_mat_x, std::vector<std::vector<float>> fp32_mat_y) {
    std::vector<std::vector<float>> fp8_mat_x = operands::fp32_mat_downcast(fp32_mat_x);
    std::vector<std::vector<float>> fp8_mat_y = operands::fp32_mat_downcast(fp32_mat_y);

    if (fp8_mat_x.empty() || fp8_mat_y.empty()) {
        throw std::invalid_argument("Empty input matrix");
    }

    int rows_x = static_cast<int>(fp8_mat_x.size());
    int cols_x = static_cast<int>(fp8_mat_x[0].size());
    int rows_y = static_cast<int>(fp8_mat_y.size());
    int cols_y = static_cast<int>(fp8_mat_y[0].size());

    if (cols_x != rows_y) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication");
    }

    std::vector<std::vector<float>> res_mat(rows_x, std::vector<float>(cols_y, 0.0f));

    #pragma omp parallel for collapse(2) num_threads(8)
    for (int i = 0; i < rows_x; i++) {
        for (int j = 0; j < cols_y; j++) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < cols_x; k++) {
                sum += operands::lmul_single(fp8_mat_x[i][k], fp8_mat_y[k][j]);
            }
            res_mat[i][j] = sum;
        }
    }

    return res_mat;
}
