#ifndef SCENE_H
#define SCENE_H

struct Hittable;

struct Scene
{
	int num_hittables;
	Hittable* hittables;
};

#endif
