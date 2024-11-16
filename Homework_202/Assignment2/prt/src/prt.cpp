#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/ray.h>
#include <filesystem/resolver.h>
#include <sh/spherical_harmonics.h>
#include <sh/default_image.h>
#include <Eigen/Core>
#include <fstream>
#include <random>
#include <stb_image.h>

NORI_NAMESPACE_BEGIN

namespace ProjEnv
{
    std::vector<std::unique_ptr<float[]>>
    LoadCubemapImages(const std::string &cubemapDir, int &width, int &height,
                      int &channel)
    {
        std::vector<std::string> cubemapNames{"negx.jpg", "posx.jpg", "posy.jpg",
                                              "negy.jpg", "posz.jpg", "negz.jpg"};
        std::vector<std::unique_ptr<float[]>> images(6);
        for (int i = 0; i < 6; i++)
        {
            std::string filename = cubemapDir + "/" + cubemapNames[i];
            int w, h, c;
            float *image = stbi_loadf(filename.c_str(), &w, &h, &c, 3);
            if (!image)
            {
                std::cout << "Failed to load image: " << filename << std::endl;
                exit(-1);
            }
            if (i == 0)
            {
                width = w;
                height = h;
                channel = c;
            }
            else if (w != width || h != height || c != channel)
            {
                std::cout << "Dismatch resolution for 6 images in cubemap" << std::endl;
                exit(-1);
            }
            images[i] = std::unique_ptr<float[]>(image);
            int index = (0 * 128 + 0) * channel;
            // std::cout << images[i][index + 0] << "\t" << images[i][index + 1] << "\t"
            //           << images[i][index + 2] << std::endl;
        }
        return images;
    }

    const Eigen::Vector3f cubemapFaceDirections[6][3] = 
    {
        {{0, 0, 1}, {0, -1, 0}, {-1, 0, 0}},  // negx
        {{0, 0, 1}, {0, -1, 0}, {1, 0, 0}},   // posx
        {{1, 0, 0}, {0, 0, -1}, {0, -1, 0}},  // negy
        {{1, 0, 0}, {0, 0, 1}, {0, 1, 0}},    // posy
        {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}}, // negz
        {{1, 0, 0}, {0, -1, 0}, {0, 0, 1}},   // posz
    };

    float CalcPreArea(const float &x, const float &y)
    {
        return std::atan2(x * y, std::sqrt(x * x + y * y + 1.0));
    }

    float CalcArea( const float &u_, 
                    const float &v_, 
                    const int &width,
                    const int &height)
    {
        // transform from [0..res - 1] to [- (1 - 1 / res) .. (1 - 1 / res)]
        // ( 0.5 is for texel center addressing)
        float u = (2.0 * (u_ + 0.5) / width) - 1.0;
        float v = (2.0 * (v_ + 0.5) / height) - 1.0;

        // shift from a demi texel, mean 1.0 / size  with u and v in [-1..1]
        float invResolutionW = 1.0 / width;
        float invResolutionH = 1.0 / height;

        // u and v are the -1..1 texture coordinate on the current face.
        // get projected area for this texel
        float x0 = u - invResolutionW;
        float y0 = v - invResolutionH;
        float x1 = u + invResolutionW;
        float y1 = v + invResolutionH;
        float angle = CalcPreArea(x0, y0) - CalcPreArea(x0, y1) -
                      CalcPreArea(x1, y0) + CalcPreArea(x1, y1);

        return angle;
    }

    // template <typename T> T ProjectSH() {}

    template <size_t SHOrder>
    std::vector<Eigen::Array3f> PrecomputeCubemapSH(const std::vector<std::unique_ptr<float[]>> &images,
                                                    const int &width, const int &height,
                                                    const int &channel)
    {
        // 预计算cubemap的采样方向cubemapDirs
        std::vector<Eigen::Vector3f> cubemapDirs;
        cubemapDirs.reserve(6 * width * height);
        for (int i = 0; i < 6; i++)
        {
            Eigen::Vector3f faceDirX = cubemapFaceDirections[i][0];
            Eigen::Vector3f faceDirY = cubemapFaceDirections[i][1];
            Eigen::Vector3f faceDirZ = cubemapFaceDirections[i][2];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float u = 2 * ((x + 0.5) / width) - 1;
                    float v = 2 * ((y + 0.5) / height) - 1;
                    Eigen::Vector3f dir = (faceDirX * u + faceDirY * v + faceDirZ).normalized();
                    cubemapDirs.push_back(dir);
                }
            }
        }

        // 预留系数数组
        constexpr int SHNum = (SHOrder + 1) * (SHOrder + 1);
        std::vector<Eigen::Array3f> SHCoeffiecents(SHNum);
        for (int i = 0; i < SHNum; i++)
            SHCoeffiecents[i] = Eigen::Array3f(0);

        float sumWeight = 0;
        for (int i = 0; i < 6; i++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    // TODO: here you need to compute light sh of each face of cubemap of each pixel
                    // TODO: 此处你需要计算每个像素下cubemap某个面的球谐系数

                    // 采样cubemap得到对应方向的入射光Le
                    Eigen::Vector3f dir = cubemapDirs[i * width * height + y * width + x];
                    int index = (y * width + x) * channel;
                    Eigen::Array3f Le(  images[i][index + 0], 
                                        images[i][index + 1],
                                        images[i][index + 2]);

                    // Edit Start
                    auto delta_w = CalcArea(x, y, width, height);
                    for (int l = 0; l <= SHOrder; l++) 
                    {
                        for (int m = -l; m <= l; m++) 
                        {
                            auto basic_sh_proj = sh::EvalSH(l, m, Eigen::Vector3d(dir.x(), dir.y(), dir.z()).normalized());
                            SHCoeffiecents[sh::GetIndex(l, m)] += Le * basic_sh_proj * delta_w;
                        }
                    }
                    // Edit End
                }
            }
        }
        return SHCoeffiecents;
    }
}

class PRTIntegrator : public Integrator
{
public:
    static constexpr int SHOrder = 2;
    static constexpr int SHCoeffLength = (SHOrder + 1) * (SHOrder + 1);

    enum class Type
    {
        Unshadowed = 0,
        Shadowed = 1,
        Interreflection = 2
    };

    PRTIntegrator(const PropertyList &props)
    {
        /* No parameters this time */
        m_SampleCount = props.getInteger("PRTSampleCount", 100);
        m_CubemapPath = props.getString("cubemap");
        auto type = props.getString("type", "unshadowed");
        if (type == "unshadowed")
        {
            m_Type = Type::Unshadowed;
        }
        else if (type == "shadowed")
        {
            m_Type = Type::Shadowed;
        }
        else if (type == "interreflection")
        {
            m_Type = Type::Interreflection;
            m_Bounce = props.getInteger("bounce", 1);
        }
        else
        {
            throw NoriException("Unsupported type: %s.", type);
        }
    }

    std::unique_ptr<std::vector<double>> computeInterreflectionSH(Eigen::MatrixXf* directTSHCoeffs, const Point3f& pos, const Normal3f& normal, const Scene* scene, int bounces)
    {
        std::unique_ptr<std::vector<double>> coeffs(new std::vector<double>());
        coeffs->assign(SHCoeffLength, 0.0);

        if (bounces > m_Bounce)
            return coeffs;

        // 在物体的某个点pos的半球面上发出采样光线，计算这些光线与场景中其他物体的相交信息
        const int sample_side = static_cast<int>(floor(sqrt(m_SampleCount)));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> rng(0.0, 1.0);
        for (int t = 0; t < sample_side; t++) 
        {
            for (int p = 0; p < sample_side; p++) 
            {
                double alpha    = (t + rng(gen)) / sample_side;
                double beta     = (p + rng(gen)) / sample_side;
                double phi      = 2.0 * M_PI * beta;
                double theta    = acos(2.0 * alpha - 1.0);

                Eigen::Array3d d = sh::ToVector(phi, theta);
                const auto wi = Vector3f(d.x(), d.y(), d.z());
                double H = wi.normalized().dot(normal);// 输入光线与法线的夹角
                
                // 记录顶点pos处，采样光线wi与场景中其他物体的相交信息
                Intersection its;

                // 如果半球面上的采样光线与其他物体相交，则这个相交点是一个次级光源
                if (H > 0.0 && scene->rayIntersect(Ray3f(pos, wi.normalized()), its))
                {
                    MatrixXf normals    = its.mesh->getVertexNormals();// 相交点所在的平面有三根法线
                    Point3f idx         = its.tri_index;// 记录了当前交点的三个三角形的序号
                    Point3f hitPos      = its.p;// 相交点的坐标
                    Vector3f bary       = its.bary;// 记录了当前交点的重心坐标
                    
                    // 对法线插值
                    Normal3f hitNormal = Normal3f(  normals.col(idx.x()).normalized() * bary.x() +
                                                    normals.col(idx.y()).normalized() * bary.y() +
                                                    normals.col(idx.z()).normalized() * bary.z()).normalized();

                    // 递归
                    auto nextBouncesCoeffs = computeInterreflectionSH(directTSHCoeffs, hitPos, hitNormal, scene, bounces + 1);

                    for (int i = 0; i < SHCoeffLength; i++)
                    {
                        // 对球谐系数插值，插值得到某个交点处新的传输系数，将所有的次级传输系数加在一起，并乘上cos值
                        auto interpolateSH = (  directTSHCoeffs->col(idx.x()).coeffRef(i) * bary.x() +
                                                directTSHCoeffs->col(idx.y()).coeffRef(i) * bary.y() +
                                                directTSHCoeffs->col(idx.z()).coeffRef(i) * bary.z());

                        (*coeffs)[i] += (interpolateSH + (*nextBouncesCoeffs)[i]) * H;
                    }
                }
            }
        }

        for (unsigned int i = 0; i < coeffs->size(); i++) 
        {
            (*coeffs)[i] /= sample_side * sample_side;
        }
        
        return coeffs;
    }

    virtual void preprocess(const Scene *scene) override
    {
        // Here only compute one mesh
        const auto mesh = scene->getMeshes()[0];
        
        // Projection environment
        auto cubePath = getFileResolver()->resolve(m_CubemapPath);
        auto lightPath = cubePath / "light.txt";
        auto transPath = cubePath / "transport.txt";

        std::ofstream lightFout(lightPath.str());
        std::ofstream fout(transPath.str());
        int width, height, channel;
        std::vector<std::unique_ptr<float[]>> images = ProjEnv::LoadCubemapImages(cubePath.str(), width, height, channel);
        
        // 1、对球谐函数投影，计算环境光的球谐函数系数，得到light.txt
        auto envCoeffs = ProjEnv::PrecomputeCubemapSH<SHOrder>(images, width, height, channel);
        m_LightCoeffs.resize(3, SHCoeffLength);
        for (int i = 0; i < envCoeffs.size(); i++)
        {
            lightFout << (envCoeffs)[i].x() << " " << (envCoeffs)[i].y() << " " << (envCoeffs)[i].z() << std::endl;
            m_LightCoeffs.col(i) = (envCoeffs)[i];
        }
        std::cout << "Computed light sh coeffs from: " << cubePath.str() << " to: " << lightPath.str() << std::endl;
        
        // 2、对球谐函数投影，计算传输项的球谐函数系数，Projection transport
        m_TransportSHCoeffs.resize(SHCoeffLength, mesh->getVertexCount());
        fout << mesh->getVertexCount() << std::endl;
        
        // 直接光照的计算（有阴影和无阴影）
        // 把每个顶点当做一个着色点，计算该顶点处的传入项 V(w)*max(0,n*w) 系数
        // 对每个顶点的所有方向采样，计算其V(w)可见性，和max(0,n*w)的投影函数
        // 每个顶点就会有一组球谐函数系数
        // 实时计算光照Lo的时候，以每个顶点为单位计算
        for (int i = 0; i < mesh->getVertexCount(); i++)
        {
            const Point3f  &v = mesh->getVertexPositions().col(i);
            const Normal3f &n = mesh->getVertexNormals().col(i);

            // phi theta 是回调函数的输入参数          
            auto shFunc = [&](double phi, double theta) -> double 
            {
                Eigen::Array3d d = sh::ToVector(phi, theta);
                const auto wi = Vector3f(d.x(), d.y(), d.z());
                // Edit Start
                double H = wi.normalized().dot(n.normalized());
                // Edit End

                // diffuse 无阴影的计算方式
                if (m_Type == Type::Unshadowed)
                {
                    // TODO: here you need to calculate unshadowed transport term of a given direction
                    // TODO: 此处你需要计算给定方向下的unshadowed传输项球谐函数值
                    return H > 0.0 ? H : 0;
                }
                // shadowed 和 Interreflection的计算方式
                else
                {
                    // TODO: here you need to calculate shadowed transport term of a given direction
                    // TODO: 此处你需要计算给定方向下的shadowed传输项球谐函数值
                    if (H > 0.0 && !scene->rayIntersect(Ray3f(v, wi.normalized()))) 
                    {
                        return H;
                    }
                    return 0;
                }
            };

            // 每个顶点处的球谐系数，个数是9组RGB（一共27个）
            auto shCoeff = sh::ProjectFunction(SHOrder, shFunc, m_SampleCount);
            for (int j = 0; j < shCoeff->size(); j++)
            {
                // Edit Start
                m_TransportSHCoeffs.col(i).coeffRef(j) = (*shCoeff)[j] / M_PI ;
                // Edit End
            }
        }

        // 间接光照的计算。多次弹射，在前面计算的基础上，继续计算
        // 在每个顶点都计算
        if (m_Type == Type::Interreflection)
        {
            // TODO: leave for bonus

            for (int i = 0; i < mesh->getVertexCount(); i++)
            {
                const Point3f& v  = mesh->getVertexPositions().col(i);// 获取当前 mesh 的第 i 个顶点
                const Normal3f& n = mesh->getVertexNormals().col(i).normalized();

                // 对所有弹射产生的次级传输系数插值，并累加，得到一个最终的次级传输系数indirectCoeffs
                auto indirectCoeffs = computeInterreflectionSH(&m_TransportSHCoeffs, v, n, scene, 1);
                for (int j = 0; j < SHCoeffLength; j++)
                {
                    m_TransportSHCoeffs.col(i).coeffRef(j) += (*indirectCoeffs)[j];
                }
                std::cout << "computing interreflection light sh coeffs, current vertex idx: " << i << " total vertex idx: " << mesh->getVertexCount() << std::endl;
            }
        }

        // Save in face format
        for (int f = 0; f < mesh->getTriangleCount(); f++)
        {
            const MatrixXu &F = mesh->getIndices();
            uint32_t idx0 = F(0, f), idx1 = F(1, f), idx2 = F(2, f);
            for (int j = 0; j < SHCoeffLength; j++)
            {
                fout << m_TransportSHCoeffs.col(idx0).coeff(j) << " ";
            }
            fout << std::endl;

            for (int j = 0; j < SHCoeffLength; j++)
            {
                fout << m_TransportSHCoeffs.col(idx1).coeff(j) << " ";
            }
            fout << std::endl;

            for (int j = 0; j < SHCoeffLength; j++)
            {
                fout << m_TransportSHCoeffs.col(idx2).coeff(j) << " ";
            }
            fout << std::endl;
        }
        std::cout << "Computed SH coeffs"
                  << " to: " << transPath.str() << std::endl;
    }

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
    {
        Intersection its;
        if (!scene->rayIntersect(ray, its))
            return Color3f(0.0f);

        const Eigen::Matrix<Vector3f::Scalar, SHCoeffLength, 1> sh0 = m_TransportSHCoeffs.col(its.tri_index.x()),
                                                                sh1 = m_TransportSHCoeffs.col(its.tri_index.y()),
                                                                sh2 = m_TransportSHCoeffs.col(its.tri_index.z());

        const Eigen::Matrix<Vector3f::Scalar, SHCoeffLength, 1> rL = m_LightCoeffs.row(0), 
                                                                gL = m_LightCoeffs.row(1), 
                                                                bL = m_LightCoeffs.row(2);

        Color3f c0 = Color3f(rL.dot(sh0), gL.dot(sh0), bL.dot(sh0)),
                c1 = Color3f(rL.dot(sh1), gL.dot(sh1), bL.dot(sh1)),
                c2 = Color3f(rL.dot(sh2), gL.dot(sh2), bL.dot(sh2));

        const Vector3f &bary = its.bary;
        Color3f c = bary.x() * c0 + bary.y() * c1 + bary.z() * c2;
        return c;
    }

    std::string toString() const
    {
        return "PRTIntegrator[]";
    }

private:
    Type m_Type;
    int m_Bounce = 1;
    int m_SampleCount = 100;
    std::string m_CubemapPath;
    Eigen::MatrixXf m_TransportSHCoeffs;
    Eigen::MatrixXf m_LightCoeffs;
};

NORI_REGISTER_CLASS(PRTIntegrator, "prt");
NORI_NAMESPACE_END