#ifndef VEC3H
#define VEC3H

#include <iostream>
#include <math.h>
#include <stdlib.h>


class vec3
{
	public:
		__host__ __device__ vec3() {}
		__host__ __device__ vec3(float e0, float e1, float e2) {e[0] = e0; e[1] = e1; e[2] = e2;}
		__host__ __device__ inline float x() const {return e[0];}
		__host__ __device__ inline float y() const {return e[1];}
		__host__ __device__ inline float z() const {return e[2];}
		__host__ __device__ inline float r() const {return e[0];}
		__host__ __device__ inline float g() const {return e[1];}
		__host__ __device__ inline float b() const {return e[2];}

		__host__ __device__ inline const vec3& operator+() const {return *this;}
		__host__ __device__ inline vec3 operator-() const {return vec3(-e[0],-e[1],-e[2]);}
		__host__ __device__ inline float operator[](int i) const {return e[i];}
		__host__ __device__ inline float& operator[](int i) {return e[i];}

		__host__ __device__ inline vec3& operator+=(const vec3 &v2);
		__host__ __device__ inline vec3& operator-=(const vec3 &v2);
		__host__ __device__ inline vec3& operator*=(const vec3 &v2);
		__host__ __device__ inline vec3& operator/=(const vec3 &v2);
		__host__ __device__ inline vec3& operator+=(const float t);
		__host__ __device__ inline vec3& operator-=(const float t);
		__host__ __device__ inline vec3& operator*=(const float t);
		__host__ __device__ inline vec3& operator/=(const float t);

		__host__ __device__ inline float length() const {return sqrt(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]);}
		__host__ __device__ inline float squared_length() const {return e[0]*e[0]+e[1]*e[1]+e[2]*e[2];}
		__host__ __device__ inline void make_unit_vector();

		float e[3];
};

inline std::istream& operator>>(std::istream &is, vec3 &t)
{
	is>>t.e[0]>>t.e[1]>>t.e[2];
	return is;
}

inline std::ostream& operator>>(std::ostream &os, vec3 &t)
{
	os<<t.e[0]<<" "<<t.e[1]<<" "<<t.e[2];
	return os;
}

__host__ __device__ inline void vec3::make_unit_vector()
{
	float d = 1.0 / sqrt(e[0]*e[0]+e[1]*e[1]+e[2]*e[2]);
	e[0] *= d; e[1] *= d; e[2] *= d;
}

__host__ __device__ inline vec3 operator+(const vec3 &a, const vec3& b)
{
	return vec3(a.e[0]+b.e[0], a.e[1]+b.e[1], a.e[2]+b.e[2]);
}

__host__ __device__ inline vec3 operator+(const vec3 &a, float t)
{
	return vec3(a.e[0]+t, a.e[1]+t, a.e[2]+t);
}

__host__ __device__ inline vec3 operator+(float t, const vec3 &a)
{
	return vec3(a.e[0]+t, a.e[1]+t, a.e[2]+t);
}

__host__ __device__ inline vec3 operator-(const vec3 &a, const vec3& b)
{
	return vec3(a.e[0]-b.e[0], a.e[1]-b.e[1], a.e[2]-b.e[2]);
}


__host__ __device__ inline vec3 operator-(const vec3 &a, float t)
{
	return vec3(a.e[0]-t, a.e[1]-t, a.e[2]-t);
}

__host__ __device__ inline vec3 operator-(float t, const vec3 &a)
{
	return vec3(t-a.e[0], t-a.e[1], t-a.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &a, const vec3& b)
{
	return vec3(a.e[0]*b.e[0], a.e[1]*b.e[1], a.e[2]*b.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &a, float t)
{
	return vec3(t*a.e[0], t*a.e[1], t*a.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &a)
{
	return vec3(t*a.e[0], t*a.e[1], t*a.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &a, const vec3& b)
{
	return vec3(a.e[0]/b.e[0], a.e[1]/b.e[1], a.e[2]/b.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &a, float t)
{
	return vec3(a.e[0]/t, a.e[1]/t, a.e[2]/t);
}

__host__ __device__ inline vec3 operator/(float t, const vec3 &a)
{
	return vec3(t/a.e[0], t/a.e[1], t/a.e[2]);
}

__host__ __device__ inline float dot(const vec3 &a, const vec3 &b)
{
	return a.e[0]*b.e[0]
		 + a.e[1]*b.e[1]
		 + a.e[2]*b.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &a, const vec3 &b)
{
	return vec3(a.e[1] * b.e[2] - a.e[2] * b.e[1],
				a.e[2] * b.e[0] - a.e[0] * b.e[2],
				a.e[0] * b.e[1] - a.e[1] * b.e[0]);
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3 &v)
{
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3 &v)
{
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];
	return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3 &v)
{
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];
	return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3 &v)
{
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];
	return *this;
}

__host__ __device__ inline vec3& vec3::operator+=(const float t)
{
	e[0] += t;
	e[1] += t;
	e[2] += t;
	return *this;
}


__host__ __device__ inline vec3& vec3::operator-=(const float t)
{
	e[0] -= t;
	e[1] -= t;
	e[2] -= t;
	return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}


__host__ __device__ inline vec3& vec3::operator/=(const float t)
{
	e[0] /= t;
	e[1] /= t;
	e[2] /= t;
	return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 v)
{
	return v / v.length();
}

#endif
