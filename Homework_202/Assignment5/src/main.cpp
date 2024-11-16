#include <fstream>
#include <string>

#include "denoiser.h"
#include "util/image.h"
#include "util/mathutil.h"

std::vector<Matrix4x4> ReadMatrix(const std::string &filename) 
{
    std::ifstream is;
    is.open(filename, std::ios::binary);
    CHECK(is.is_open());
    
    int shapeNum;
    is.read(reinterpret_cast<char *>(&shapeNum), sizeof(int));
    std::vector<Matrix4x4> matrix(shapeNum + 2);
    
    for (int i = 0; i < shapeNum + 2; i++)
    {
        is.read(reinterpret_cast<char *>(&matrix[i]), sizeof(Matrix4x4));
    }
    is.close();
    
    return matrix;
}

FrameInfo LoadFrameInfo(const filesystem::path &inputDir, const int &idx) 
{
    Buffer2D<Float3> beauty   = ReadFloat3Image((inputDir / ("beauty_"    + std::to_string(idx) + ".exr")).str());
    Buffer2D<Float3> normal   = ReadFloat3Image((inputDir / ("normal_"    + std::to_string(idx) + ".exr")).str());
    Buffer2D<Float3> position = ReadFloat3Image((inputDir / ("position_"  + std::to_string(idx) + ".exr")).str());
    Buffer2D<float> depth     = ReadFloatImage(( inputDir / ("depth_"     + std::to_string(idx) + ".exr")).str());
    Buffer2D<float> id        = ReadFloatImage(( inputDir / ("ID_"        + std::to_string(idx) + ".exr")).str());
    
    // 保存了多个矩阵。m_matrix[i] 表示标号为 i 的物体从物体坐标系到世界坐标系的矩阵。
    // 此外，m_matrix 中的倒数第 2 个和倒数第 1 个分别为，
    // 世界坐标系到摄像机坐标系、世界坐标系到屏幕坐标系 ([0, W)×[0,H)) 的矩阵。
    std::vector<Matrix4x4> matrix   = ReadMatrix((inputDir / ("matrix_" + std::to_string(idx) + ".mat")).str());

    // 一帧的信息，512X512的一张图上，各个像素点的 信息
    FrameInfo frameInfo = {beauty, depth, normal, position, id, matrix};
    
    return frameInfo;
}

void Denoise(const filesystem::path &inputDir, 
             const filesystem::path &outputDir,
             const int &frameNum) 
{
    Denoiser denoiser;

    // 对所有读取的帧降噪
    for (int i = 0; i < frameNum; i++) 
    {
        std::cout << "Frame: " << i << std::endl;
        
        FrameInfo frameInfo    = LoadFrameInfo(inputDir, i);
        Buffer2D<Float3> image = denoiser.ProcessFrame(frameInfo);// 降噪后的image
        std::string filename   = (outputDir / ("result_" + std::to_string(i) + ".exr")).str();
        WriteFloat3Image(image, filename);
    }
}

int main() 
{
    // Box
    filesystem::path inputDir( "examples/box/input");
    filesystem::path outputDir("examples/box/output");
    int frameNum = 20;

    /*
    // Pink room
    filesystem::path inputDir("examples/pink-room/input");
    filesystem::path outputDir("examples/pink-room/output");
    int frameNum = 80;
    */

    Denoise(inputDir, outputDir, frameNum);
    return 0;
}
