// Narrow compatibility layer for a separately built Box2D 2.4.1 comparator.
// The work argument means velocity iterations here, not temporal substeps.
#pragma once
#include <box2d/box2d.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

#define B2_MAX_POLYGON_VERTICES b2_maxPolygonVertices
using b2WorldId = b2World*;
using b2BodyId = b2Body*;
using b2ShapeId = b2Fixture*;
using b2Polygon = b2PolygonShape;
using b2Circle = b2CircleShape;
constexpr int b2_circleShape = b2Shape::e_circle;
struct RigidWorldDef {
    b2Vec2 gravity{0, -9.81f};
    float contactHertz=10, contactDampingRatio=1, maxContactPushSpeed=1;
    float restitutionThreshold=0, maximumLinearSpeed=500;
    bool enableSleep=false, enableContinuous=true;
};
struct RigidBodyDef {
    b2BodyType type=b2_staticBody; bool fixedRotation=false, isBullet=false, enableSleep=false;
    b2Vec2 position{0,0}, linearVelocity{0,0}; b2Rot rotation{0}; float angularVelocity=0;
    void* userData=nullptr;
};
struct RigidMaterial { float friction=.3f, restitution=0, rollingResistance=0; };
struct RigidShapeDef { float density=1; RigidMaterial material; };
struct RigidPoint { float separation=0; };
struct RigidManifold { int pointCount=0; RigidPoint points[2]; };
struct RigidMassData { float mass=0; b2Vec2 center{0,0}; float rotationalInertia=0; };
struct b2ContactData { b2ShapeId shapeIdA=nullptr, shapeIdB=nullptr; };
struct b2Hull { int count=0; b2Vec2 points[B2_MAX_POLYGON_VERTICES]; };
inline RigidWorldDef b2DefaultWorldDef() { return {}; }
inline RigidBodyDef b2DefaultBodyDef() { return {}; }
inline RigidShapeDef b2DefaultShapeDef() { return {}; }
inline b2Rot b2MakeRot(float a) { return b2Rot(a); }
inline float b2Length(b2Vec2 v) { return v.Length(); }
inline b2Vec2 b2Add(b2Vec2 a,b2Vec2 b) { return a+b; }
inline b2Vec2 b2MulSV(float s,b2Vec2 v) { return s*v; }
inline b2WorldId b2CreateWorld(const RigidWorldDef* def) {
    auto world=new b2World(def->gravity); world->SetAllowSleeping(def->enableSleep);
    world->SetContinuousPhysics(def->enableContinuous); return world;
}
inline void b2DestroyWorld(b2WorldId world) { delete world; }
inline void b2World_Step(b2WorldId world, float dt, int iterations) { world->Step(dt, iterations, 3); }
inline b2BodyId b2CreateBody(b2WorldId world, const RigidBodyDef* d) {
    b2BodyDef def; def.type=d->type; def.position=d->position; def.angle=d->rotation.GetAngle();
    def.linearVelocity=d->linearVelocity; def.angularVelocity=d->angularVelocity;
    def.fixedRotation=d->fixedRotation; def.bullet=d->isBullet; def.allowSleep=d->enableSleep;
    def.userData.pointer=reinterpret_cast<uintptr_t>(d->userData); return world->CreateBody(&def);
}
inline b2Hull b2ComputeHull(const b2Vec2* v, int count) {
    b2PolygonShape p; p.Set(v,count); b2Hull h; h.count=p.m_count;
    for(int i=0;i<h.count;++i) h.points[i]=p.m_vertices[i];
    return h;
}
inline bool b2ValidateHull(const b2Hull* h) { return h->count>=3; }
inline b2Polygon b2MakePolygon(const b2Hull* h, float radius) {
    b2Polygon p; p.Set(h->points,h->count); p.m_radius=radius; return p;
}
inline RigidMassData b2ComputePolygonMass(const b2Polygon* p,float density) {
    b2MassData md; p->ComputeMass(&md,density); return {md.mass,md.center,md.I};
}
inline void b2Body_SetMassData(b2BodyId b,RigidMassData data) {
    b2MassData md; md.mass=data.mass; md.center=data.center;
    md.I=data.rotationalInertia+data.mass*b2Dot(data.center,data.center); b->SetMassData(&md);
}
inline b2ShapeId b2CreatePolygonShape(b2BodyId body, const RigidShapeDef* d, const b2Polygon* p) {
    if(d->material.rollingResistance!=0) throw std::runtime_error("Block comparator has no rolling resistance model");
    b2FixtureDef def; def.shape=p; def.density=d->density; def.friction=d->material.friction;
    def.restitution=d->material.restitution; def.restitutionThreshold=0; return body->CreateFixture(&def);
}
inline b2ShapeId b2CreateCircleShape(b2BodyId body, const RigidShapeDef* d, const b2Circle* p) {
    if(d->material.rollingResistance!=0) throw std::runtime_error("Block comparator has no rolling resistance model");
    b2FixtureDef def; def.shape=p; def.density=d->density; def.friction=d->material.friction;
    def.restitution=d->material.restitution; def.restitutionThreshold=0; return body->CreateFixture(&def);
}
inline int b2Shape_GetType(b2ShapeId s) { return s->GetType(); }
inline b2Circle b2Shape_GetCircle(b2ShapeId s) { return *static_cast<const b2CircleShape*>(s->GetShape()); }
inline void b2Body_SetLinearVelocity(b2BodyId b,b2Vec2 v) { b->SetLinearVelocity(v); }
inline void b2Body_SetAngularVelocity(b2BodyId b,float w) { b->SetAngularVelocity(w); }
inline b2Vec2 b2Body_GetLinearVelocity(b2BodyId b) { return b->GetLinearVelocity(); }
inline float b2Body_GetAngularVelocity(b2BodyId b) { return b->GetAngularVelocity(); }
inline b2Vec2 b2Body_GetWorldCenterOfMass(b2BodyId b) { return b->GetWorldCenter(); }
inline b2Rot b2Body_GetRotation(b2BodyId b) { return b->GetTransform().q; }
inline b2Transform b2Body_GetTransform(b2BodyId b) { return b->GetTransform(); }
inline float b2Body_GetMass(b2BodyId b) { return b->GetMass(); }
inline float b2Body_GetRotationalInertia(b2BodyId b) {
    return std::max(0.f,b->GetInertia()-b->GetMass()*b2Dot(b->GetLocalCenter(),b->GetLocalCenter()));
}
inline void* b2Body_GetUserData(b2BodyId b) { return reinterpret_cast<void*>(b->GetUserData().pointer); }
inline b2BodyId b2Shape_GetBody(b2ShapeId s) { return s->GetBody(); }
inline b2Polygon b2Shape_GetPolygon(b2ShapeId s) { return *static_cast<const b2PolygonShape*>(s->GetShape()); }
inline uint64_t b2StoreShapeId(b2ShapeId s) { return reinterpret_cast<uintptr_t>(s); }
inline int b2Body_GetContactCapacity(b2BodyId b) {
    int n=0; for(auto edge=b->GetContactList();edge;edge=edge->next) ++n; return n;
}
inline int b2Body_GetContactData(b2BodyId b,b2ContactData* data,int capacity) {
    int n=0;
    for(auto edge=b->GetContactList();edge && n<capacity;edge=edge->next)
        data[n++]={edge->contact->GetFixtureA(),edge->contact->GetFixtureB()};
    return n;
}
inline RigidManifold b2CollidePolygons(const b2Polygon* a,b2Transform xa,const b2Polygon* b,b2Transform xb) {
    b2Manifold m; b2CollidePolygons(&m,a,xa,b,xb); b2WorldManifold w;
    w.Initialize(&m,xa,a->m_radius,xb,b->m_radius); RigidManifold out; out.pointCount=m.pointCount;
    for(int i=0;i<m.pointCount;++i) out.points[i].separation=w.separations[i];
    return out;
}
inline RigidManifold rigidCollideShapes(b2ShapeId a,b2Transform xa,b2ShapeId b,b2Transform xb) {
    const auto* sa=a->GetShape(); const auto* sb=b->GetShape(); b2Manifold m;
    if(sa->GetType()==b2Shape::e_circle && sb->GetType()==b2Shape::e_circle)
        b2CollideCircles(&m,static_cast<const b2Circle*>(sa),xa,static_cast<const b2Circle*>(sb),xb);
    else if(sa->GetType()==b2Shape::e_polygon && sb->GetType()==b2Shape::e_circle)
        b2CollidePolygonAndCircle(&m,static_cast<const b2Polygon*>(sa),xa,static_cast<const b2Circle*>(sb),xb);
    else if(sa->GetType()==b2Shape::e_circle && sb->GetType()==b2Shape::e_polygon)
        return rigidCollideShapes(b,xb,a,xa);
    else b2CollidePolygons(&m,static_cast<const b2Polygon*>(sa),xa,static_cast<const b2Polygon*>(sb),xb);
    b2WorldManifold w; w.Initialize(&m,xa,sa->m_radius,xb,sb->m_radius);
    RigidManifold out; out.pointCount=m.pointCount;
    for(int i=0;i<m.pointCount;++i) out.points[i].separation=w.separations[i];
    return out;
}
