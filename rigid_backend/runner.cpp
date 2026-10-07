// Headless benchmark adapter. Box2D retains its contact caches across fidelity changes.
#ifdef RIGID_BLOCK_BACKEND
#include "compat2.h"
#else
#include <box2d/box2d.h>
using RigidWorldDef = b2WorldDef;
using RigidBodyDef = b2BodyDef;
using RigidShapeDef = b2ShapeDef;
using RigidManifold = b2Manifold;
using RigidMassData = b2MassData;
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>
#include <unordered_map>

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
template<class T> static T read() {
    T value;
    if (!(std::cin >> value)) throw std::runtime_error("Truncated scene input");
    return value;
}
#ifndef RIGID_BLOCK_BACKEND
// Hull authoring welds at a fixed metre-scale slop. Already ordered strictly
// convex input can contain legitimate corners below that slop. Construct with
// the public helper at a larger authoring scale, then restore the exact core;
// normals are dimensionless. This changes neither world units nor solver slop.
static b2Polygon orderedPolygon(const b2Vec2* input, int count, float radius) {
    b2Vec2 origin{0,0};
    for(int i=0;i<count;++i) origin=b2Add(origin,input[i]);
    origin=b2MulSV(1.f/count,origin);
    double area=0, minimum=1e30;
    for(int i=0;i<count;++i) {
        const auto& a=input[i]; const auto& b=input[(i+1)%count];
        area+=double(a.x)*b.y-double(a.y)*b.x;
    }
    float winding=area>0 ? 1.f : -1.f;
    for(int i=0;i<count;++i) {
        const auto& a=input[i]; const auto& b=input[(i+1)%count];
        double dx=double(b.x)-a.x, dy=double(b.y)-a.y, length=std::hypot(dx,dy);
        if(length<.009999) throw std::runtime_error("Polygon edge below supported scale");
        minimum=std::min(minimum,length*.5);
        for(int j=0;j<count;++j) if(j!=i && j!=(i+1)%count) {
            double distance=winding*(dx*(double(input[j].y)-a.y)-dy*(double(input[j].x)-a.x))/length;
            if(!(distance>1e-9)) throw std::runtime_error("Strict ordered convex polygon required after Float32 conversion");
            minimum=std::min(minimum,distance);
        }
    }
    float scale=float(std::max(1.,.04/minimum));
    if(!std::isfinite(scale) || scale>1e6f) throw std::runtime_error("Polygon construction is ill-conditioned");
    b2Vec2 vertices[B2_MAX_POLYGON_VERTICES];
    for(int i=0;i<count;++i) vertices[i]=b2MulSV(scale,b2Sub(input[i],origin));
    auto hull=b2ComputeHull(vertices,count);
    if(hull.count!=count || !b2ValidateHull(&hull)) throw std::runtime_error("Convex ordered polygon construction failed");
    auto polygon=b2MakePolygon(&hull,0);
    for(int i=0;i<count;++i) polygon.vertices[i]=b2Add(origin,b2MulSV(1.f/scale,polygon.vertices[i]));
    polygon.centroid=b2Add(origin,b2MulSV(1.f/scale,polygon.centroid));
    polygon.radius=radius;
    return polygon;
}
#endif
struct Body {
    b2BodyId id;
    bool dynamic;
    bool kinematic;
    float extent = 1e20f, radius = 0, mass = 0, inertia = 0;
    std::array<double,3> prescribedPose{};
};
struct BoundaryFixture { b2ShapeId id; int body; float radius; std::vector<b2Vec2> vertices; };
struct BoundaryFilter {
    std::vector<BoundaryFixture> fixtures;
    std::unordered_map<uint64_t,size_t> index;
    std::vector<std::vector<size_t>> bodyFixtures;
    long long removed=0;
    bool internal(b2ShapeId shape,b2Vec2 surface,b2Vec2 outward) const {
        auto found=index.find(b2StoreShapeId(shape));
        if(found==index.end()) return false;
        const auto& fixture=fixtures[found->second];
        if(bodyFixtures[fixture.body].size()<2) return false;
        auto transform=b2Body_GetTransform(b2Shape_GetBody(shape));
        // Step just outside the fixture's unrounded core in the reaction direction.
        b2Vec2 sample=b2Add(surface,b2MulSV(1e-5f-fixture.radius,outward));
#ifdef RIGID_BLOCK_BACKEND
        sample=b2MulT(transform,sample);
#else
        sample=b2InvTransformPoint(transform,sample);
#endif
        for(auto otherIndex:bodyFixtures[fixture.body]) {
            if(otherIndex==found->second) continue;
            const auto& polygon=fixtures[otherIndex].vertices;
            bool inside=true;
            for(size_t i=0;i<polygon.size();++i) {
                auto a=polygon[i], b=polygon[(i+1)%polygon.size()];
                double dx=double(b.x)-a.x,dy=double(b.y)-a.y;
                double side=(dx*(double(sample.y)-a.y)-dy*(double(sample.x)-a.x))/std::hypot(dx,dy);
                if(side<=1e-7) { inside=false; break; }
            }
            if(inside) return true;
        }
        return false;
    }
};
#ifdef RIGID_BLOCK_BACKEND
struct BoundaryListener : b2ContactListener {
    BoundaryFilter* filter;
    double boundaryWork=0,absoluteBoundaryWork=0,frictionImpulseAbs=0;
    long long boundaryImpulsePoints=0;
    explicit BoundaryListener(BoundaryFilter* f):filter(f){}
    void PreSolve(b2Contact* contact,const b2Manifold*) override {
        if(!filter)return;
        auto* m=contact->GetManifold(); b2WorldManifold world;
        contact->GetWorldManifold(&world);
        int kept=0;
        for(int i=0;i<m->pointCount;++i) {
            auto n=world.normal,p=world.points[i]; float separation=world.separations[i];
            bool hide=filter->internal(contact->GetFixtureA(),p+(-.5f*separation)*n,n) ||
                      filter->internal(contact->GetFixtureB(),p+(.5f*separation)*n,-n);
            if(hide) ++filter->removed; else m->points[kept++]=m->points[i];
        }
        m->pointCount=kept;
    }
    void PostSolve(b2Contact* contact,const b2ContactImpulse* impulses) override {
        for(int i=0;i<impulses->count;i++)frictionImpulseAbs+=std::abs(double(impulses->tangentImpulses[i]));
        auto* a=contact->GetFixtureA()->GetBody();auto* b=contact->GetFixtureB()->GetBody();
        const bool aBoundary=a->GetType()==b2_kinematicBody&&b->GetType()==b2_dynamicBody;
        const bool bBoundary=b->GetType()==b2_kinematicBody&&a->GetType()==b2_dynamicBody;
        if(!aBoundary&&!bBoundary)return;
        b2WorldManifold world;contact->GetWorldManifold(&world);
        const auto tangent=b2Cross(world.normal,1.f);
        for(int i=0;i<impulses->count;i++){
            auto impulse=impulses->normalImpulses[i]*world.normal+impulses->tangentImpulses[i]*tangent;
            const auto velocity=(aBoundary?a:b)->GetLinearVelocityFromWorldPoint(world.points[i]);
            const double work=(aBoundary?1.:-1.)*(double(impulse.x)*velocity.x+double(impulse.y)*velocity.y);
            boundaryWork+=work;absoluteBoundaryWork+=std::abs(work);boundaryImpulsePoints++;
        }
    }
};
#else
static bool filterBoundary(b2ShapeId a,b2ShapeId b,b2Manifold* m,void* context) {
    auto* filter=static_cast<BoundaryFilter*>(context); int kept=0;
    for(int i=0;i<m->pointCount;++i) {
        auto n=m->normal,p=m->points[i].point; float separation=m->points[i].separation;
        bool hide=filter->internal(a,b2Add(p,b2MulSV(-.5f*separation,n)),n) ||
                  filter->internal(b,b2Add(p,b2MulSV(.5f*separation,n)),b2MulSV(-1,n));
        if(hide) ++filter->removed; else m->points[kept++]=m->points[i];
    }
    m->pointCount=kept;
    return kept>0;
}
#endif
struct Command { int frame, body; b2Vec2 velocity; float omega; };
#ifndef RIGID_BLOCK_BACKEND
static RigidManifold rigidCollideShapes(b2ShapeId a,b2Transform xa,b2ShapeId b,b2Transform xb) {
    bool ca=b2Shape_GetType(a)==b2_circleShape, cb=b2Shape_GetType(b)==b2_circleShape;
    if(ca && cb) { auto sa=b2Shape_GetCircle(a), sb=b2Shape_GetCircle(b); return b2CollideCircles(&sa,xa,&sb,xb); }
    if(!ca && cb) { auto sa=b2Shape_GetPolygon(a); auto sb=b2Shape_GetCircle(b); return b2CollidePolygonAndCircle(&sa,xa,&sb,xb); }
    if(ca && !cb) return rigidCollideShapes(b,xb,a,xa);
    auto sa=b2Shape_GetPolygon(a), sb=b2Shape_GetPolygon(b); return b2CollidePolygons(&sa,xa,&sb,xb);
}
#endif
struct Features { float travel = 0, penetration = 0, massRatio = 1; int contacts = 0, island = 0; };

static Features features(const std::vector<Body>& bodies, float dt) {
    Features f;
    std::vector<int> parent(bodies.size());
    std::iota(parent.begin(), parent.end(), 0);
    auto root = [&parent](int i) { while (parent[i] != i) i = parent[i]; return i; };
    std::set<std::pair<uint64_t, uint64_t>> seen;
    for (const Body& body : bodies) {
        if (!body.dynamic && !body.kinematic) continue;
        b2Vec2 v = b2Body_GetLinearVelocity(body.id);
        float w = b2Body_GetAngularVelocity(body.id);
        f.travel = std::max(f.travel, dt * (b2Length(v) + std::abs(w) * body.radius) / body.extent);
        std::vector<b2ContactData> contacts(b2Body_GetContactCapacity(body.id));
        int count = b2Body_GetContactData(body.id, contacts.data(), int(contacts.size()));
        for (int j = 0; j < count; ++j) {
            const auto& contact = contacts[j];
            uint64_t sa = b2StoreShapeId(contact.shapeIdA), sb = b2StoreShapeId(contact.shapeIdB);
            auto key = std::make_pair(std::min(sa, sb), std::max(sa, sb));
            if (!seen.insert(key).second) continue;
            b2BodyId a = b2Shape_GetBody(contact.shapeIdA), b = b2Shape_GetBody(contact.shapeIdB);
            int ia = int(reinterpret_cast<intptr_t>(b2Body_GetUserData(a))) - 1;
            int ib = int(reinterpret_cast<intptr_t>(b2Body_GetUserData(b))) - 1;
            RigidManifold m = rigidCollideShapes(contact.shapeIdA, b2Body_GetTransform(a), contact.shapeIdB, b2Body_GetTransform(b));
            if (!m.pointCount) continue;
            ++f.contacts;
            float extent = std::min(bodies[ia].extent, bodies[ib].extent);
            for (int k = 0; k < m.pointCount; ++k)
                f.penetration = std::max(f.penetration, std::max(0.f, -m.points[k].separation) / extent);
            if (bodies[ia].dynamic && bodies[ib].dynamic) parent[root(ia)] = root(ib);
        }
    }
    std::vector<int> sizes(bodies.size(), 0);
    std::vector<float> minMass(bodies.size(), 1e20f), maxMass(bodies.size(), 0);
    for (int i = 0; i < int(bodies.size()); ++i) {
        if (!bodies[i].dynamic) continue;
        int r = root(i);
        ++sizes[r]; minMass[r] = std::min(minMass[r], bodies[i].mass); maxMass[r] = std::max(maxMass[r], bodies[i].mass);
        f.island = std::max(f.island, sizes[r]);
        f.massRatio = std::max(f.massRatio, maxMass[r] / minMass[r]);
    }
    return f;
}

static void snapshot(std::ostream& out, const std::vector<Body>& bodies) {
    out << '['; bool comma = false;
    for (const Body& body : bodies) {
        if (!body.dynamic) continue;
        if (comma) out << ',';
        comma = true;
        b2Vec2 p = b2Body_GetWorldCenterOfMass(body.id), v = b2Body_GetLinearVelocity(body.id);
        b2Rot q = b2Body_GetRotation(body.id);
        out << '[' << p.x << ',' << p.y << ',' << std::atan2(q.s, q.c) << ',' << v.x << ',' << v.y << ','
            << b2Body_GetAngularVelocity(body.id) << ']';
    }
    out << ']';
}

int main() {
  try {
    if (read<std::string>() != "rigid-v2") throw std::runtime_error("Rebuild backend: expected rigid-v2 scene protocol");
    float dt = read<float>(); int frames = read<int>();
    RigidWorldDef wd = b2DefaultWorldDef();
    wd.gravity = {read<float>(), read<float>()}; wd.contactHertz = read<float>();
    float skin = read<float>();
    wd.contactDampingRatio = 1.f; wd.maxContactPushSpeed = 1.f;
    wd.restitutionThreshold = 0.f; wd.maximumLinearSpeed = 500.f;
    wd.enableSleep = false; wd.enableContinuous = true;
    int adaptive = read<int>(), fixedPrimary = read<int>(), fixedSubsteps = read<int>();
    float travelThreshold = read<float>(), penetrationThreshold = read<float>();
    int islandThreshold = read<int>(); float massThreshold = read<float>();
    int dwell = read<int>(), minimum = read<int>();
    int highPrimary = read<int>(), highSubsteps = read<int>();
    if (!(dt > 0) || frames < 1 || fixedPrimary < 1 || fixedSubsteps < 1 || minimum < 0 || minimum > 3)
        throw std::runtime_error("Invalid integration settings");
    b2WorldId world = b2CreateWorld(&wd);
    int bodyCount = read<int>(); std::vector<Body> bodies;
    BoundaryFilter boundary; boundary.bodyFixtures.resize(bodyCount);
    for (int i = 0; i < bodyCount; ++i) {
        RigidBodyDef bd = b2DefaultBodyDef(); int type = read<int>();
        bd.type = type == 2 ? b2_dynamicBody : (type == 1 ? b2_kinematicBody : b2_staticBody);
        bd.fixedRotation = read<int>() != 0; bd.isBullet = read<int>() != 0;
        bd.position = {read<float>(), read<float>()}; bd.rotation = b2MakeRot(read<float>());
        bd.linearVelocity = {read<float>(), read<float>()}; bd.angularVelocity = read<float>();
        bd.userData = reinterpret_cast<void*>(intptr_t(i + 1)); bd.enableSleep = false;
        Body body{b2CreateBody(world, &bd), type == 2, type == 1};
        body.prescribedPose={bd.position.x,bd.position.y,std::atan2(bd.rotation.s,bd.rotation.c)};
        float mass = 0, inertiaOrigin = 0; b2Vec2 moment{0, 0};
        int shapeCount = read<int>();
        for (int j = 0; j < shapeCount; ++j) {
            int kind = read<int>(); RigidShapeDef sd = b2DefaultShapeDef();
            sd.density = read<float>(); sd.material.friction = read<float>();
            sd.material.restitution = read<float>(); sd.material.rollingResistance = read<float>();
            if (kind == 0) {
                float radius=read<float>(); b2Vec2 center{read<float>(),read<float>()};
                if (!(radius > 0)) throw std::runtime_error("Invalid circle radius");
                b2Circle circle;
#ifdef RIGID_BLOCK_BACKEND
                circle.m_p=center; circle.m_radius=radius;
#else
                circle.center=center; circle.radius=radius;
#endif
                float cm=sd.density*float(3.141592653589793)*radius*radius;
                mass+=cm; moment=b2Add(moment,b2MulSV(cm,center));
                inertiaOrigin+=cm*(.5f*radius*radius+b2Dot(center,center));
                body.extent=std::min(body.extent,2*radius);
                body.radius=std::max(body.radius,b2Length(center)+radius);
                b2CreateCircleShape(body.id,&sd,&circle);
                continue;
            }
            if (kind != 1) throw std::runtime_error("Unknown fixture kind");
            int count = read<int>();
            if (count < 3 || count > B2_MAX_POLYGON_VERTICES) throw std::runtime_error("Invalid polygon vertex count");
            b2Vec2 vertices[B2_MAX_POLYGON_VERTICES]; b2Vec2 lo{1e20f, 1e20f}, hi{-1e20f, -1e20f};
            for (int k = 0; k < count; ++k) {
                vertices[k] = {read<float>(), read<float>()};
                lo = b2Min(lo, vertices[k]); hi = b2Max(hi, vertices[k]);
                body.radius = std::max(body.radius, b2Length(vertices[k]));
            }
#ifdef RIGID_BLOCK_BACKEND
            b2Hull hull = b2ComputeHull(vertices, count);
            if (hull.count != count || !b2ValidateHull(&hull)) throw std::runtime_error("Convex ordered polygon required");
            b2Polygon core = b2MakePolygon(&hull, 0.f);
#else
            b2Polygon core = orderedPolygon(vertices,count,0.f);
#endif
            RigidMassData md = b2ComputePolygonMass(&core, sd.density);
            mass += md.mass; moment = b2Add(moment, b2MulSV(md.mass, md.center));
            inertiaOrigin += md.rotationalInertia;
#ifdef RIGID_BLOCK_BACKEND
            b2Polygon polygon = b2MakePolygon(&hull, skin);
#else
            b2Polygon polygon=core; polygon.radius=skin;
#endif
            auto shape=b2CreatePolygonShape(body.id, &sd, &polygon);
            BoundaryFixture fixture{shape,i,skin,{}};
            double signedArea=0;
            for(int k=0;k<count;++k) {
                fixture.vertices.push_back(vertices[k]);
                signedArea+=double(vertices[k].x)*vertices[(k+1)%count].y-double(vertices[k].y)*vertices[(k+1)%count].x;
            }
            if(signedArea<0) std::reverse(fixture.vertices.begin(),fixture.vertices.end());
            boundary.index[b2StoreShapeId(shape)]=boundary.fixtures.size();
            boundary.bodyFixtures[i].push_back(boundary.fixtures.size());
            boundary.fixtures.push_back(fixture);
            body.extent = std::min(body.extent, std::min(hi.x - lo.x, hi.y - lo.y));
        }
        if (body.dynamic && mass > 0) {
            RigidMassData md; md.mass = mass; md.center = b2MulSV(1.f / mass, moment);
            md.rotationalInertia = bd.fixedRotation ? 0.f : inertiaOrigin - mass * b2Dot(md.center, md.center);
            b2Body_SetMassData(body.id, md);
        }
        body.mass = b2Body_GetMass(body.id); body.inertia = b2Body_GetRotationalInertia(body.id);
        if (shapeCount < 1 || body.extent <= 0 || (body.dynamic && body.mass <= 0)) throw std::runtime_error("Invalid body");
        bodies.push_back(body);
    }
    int commandCount=read<int>(); std::vector<Command> commands;
    for(int i=0;i<commandCount;++i) {
        int frame=read<int>(), body=read<int>(); b2Vec2 velocity{read<float>(),read<float>()}; float omega=read<float>();
        if(frame<0 || frame>=frames || body<0 || body>=bodyCount || !bodies[body].kinematic)
            throw std::runtime_error("Invalid kinematic command");
        commands.push_back({frame,body,velocity,omega});
    }
    std::stable_sort(commands.begin(),commands.end(),[](const Command& a,const Command& b){return a.frame<b.frame;});
    int suppressInternal=0;
    // Optional diagnostic policy appended to rigid-v2; old inputs keep old behaviour.
    if(!(std::cin>>suppressInternal)) std::cin.clear();
    int positionIterations=3, analyticKinematics=0;
    if(!(std::cin>>positionIterations)) std::cin.clear();
    if(!(std::cin>>analyticKinematics)) std::cin.clear();
#ifdef RIGID_BLOCK_BACKEND
    rigidPositionIterations=positionIterations;
    BoundaryListener listener(suppressInternal?&boundary:nullptr);
    world->SetContactListener(&listener);
#else
    if(suppressInternal) {
        b2World_SetPreSolveCallback(world,filterBoundary,&boundary);
        for(const auto& fixture:boundary.fixtures) b2Shape_EnablePreSolveEvents(fixture.id,true);
    }
#endif
    size_t commandIndex=0;
    std::vector<std::vector<std::array<float,6>>> kinematicHistory;
    auto captureKinematics=[&]() {
        std::vector<std::array<float,6>> state;
        for(const auto& body:bodies) if(body.kinematic) {
            auto p=b2Body_GetWorldCenterOfMass(body.id), v=b2Body_GetLinearVelocity(body.id); auto q=b2Body_GetRotation(body.id);
            state.push_back({p.x,p.y,std::atan2(q.s,q.c),v.x,v.y,b2Body_GetAngularVelocity(body.id)});
        }
        kinematicHistory.push_back(state);
    };
    captureKinematics();
    std::cout << std::setprecision(10) << "{\"states\":["; snapshot(std::cout, bodies);
    double stepSeconds = 0, controllerSeconds = 0;
    std::vector<int> histogram(4, 0), selected; selected.reserve(frames);
    int current = minimum, downFrames = 0, switches = 0;
    long long work = 0;
    float maximumPenetration = 0;
    for (int frame = 0; frame < frames; ++frame) {
        while(commandIndex<commands.size() && commands[commandIndex].frame==frame) {
            const auto& command=commands[commandIndex++];
            b2Body_SetLinearVelocity(bodies[command.body].id,command.velocity);
            b2Body_SetAngularVelocity(bodies[command.body].id,command.omega);
        }
        int primary = fixedPrimary, substeps = fixedSubsteps;
        if (adaptive) {
            auto clock = Clock::now(); Features f = features(bodies, dt);
            int desired = f.contacts ? 1 : 0;
            if (f.travel > travelThreshold) desired = std::max(desired, 2);
            if (f.travel > 2 * travelThreshold || f.penetration > penetrationThreshold ||
                (f.contacts && (f.island >= islandThreshold || f.massRatio >= massThreshold))) desired = 3;
            desired = std::max(desired, minimum);
            int old = current;
            if (desired > current) { current = desired; downFrames = 0; }
            else if (desired < current && ++downFrames >= dwell) { current = desired; downFrames = 0; }
            else if (desired == current) downFrames = 0;
            switches += current != old;
            primary = current < 2 ? 1 : (current == 2 ? 2 : highPrimary);
            substeps = current == 0 ? 1 : (current == 1 ? 4 : (current == 2 ? 8 : highSubsteps));
            ++histogram[current]; controllerSeconds += seconds(clock);
        }
        selected.push_back(adaptive ? current : -1);
        auto clock = Clock::now();
        for (int k = 0; k < primary; ++k) {
            b2World_Step(world, dt / primary, substeps);
            if(analyticKinematics) for(auto& body:bodies) if(body.kinematic) {
                auto v=b2Body_GetLinearVelocity(body.id); auto w=b2Body_GetAngularVelocity(body.id);
                body.prescribedPose[0]+=double(v.x)*double(dt)/primary;
                body.prescribedPose[1]+=double(v.y)*double(dt)/primary;
                body.prescribedPose[2]+=double(w)*double(dt)/primary;
                b2Vec2 target{float(body.prescribedPose[0]),float(body.prescribedPose[1])};
                b2Body_SetTransform(body.id,target,b2MakeRot(float(body.prescribedPose[2])));
            }
        }
        stepSeconds += seconds(clock); work += primary * substeps;
        // This observable covers reported contact pairs, not a certificate against all missed collisions.
        Features measured = features(bodies, dt);
        maximumPenetration = std::max(maximumPenetration, measured.penetration);
        std::cout << ','; snapshot(std::cout, bodies);
        captureKinematics();
    }
#ifdef RIGID_DOUBLE_PRECISION
    std::cout << "],\"scalar_precision\":\"float64\",\"mass\":["; bool comma = false;
#else
    std::cout << "],\"scalar_precision\":\"float32\",\"mass\":["; bool comma = false;
#endif
    for (auto body : bodies) if (body.dynamic) { if (comma) std::cout << ','; comma = true; std::cout << body.mass; }
    std::cout << "],\"inertia\":["; comma = false;
    for (auto body : bodies) if (body.dynamic) { if (comma) std::cout << ','; comma = true; std::cout << body.inertia; }
    std::cout << "]";
#ifdef RIGID_BLOCK_BACKEND
    std::cout << ",\"boundary_work_J\":" << listener.boundaryWork
              << ",\"absolute_boundary_work_J\":" << listener.absoluteBoundaryWork
              << ",\"friction_impulse_abs_kg_m_s\":" << listener.frictionImpulseAbs
              << ",\"boundary_impulse_points\":" << listener.boundaryImpulsePoints;
#endif
    std::cout << ",\"step_s\":" << stepSeconds << ",\"controller_s\":" << controllerSeconds
              << ",\"reported_max_penetration_fraction\":" << maximumPenetration
              << ",\"internal_contact_points_removed\":" << boundary.removed
              << ",\"solver_work_total\":" << work << ",\"switches\":" << switches << ",\"level_frames\":[";
    for (int i = 0; i < 4; ++i) std::cout << (i ? "," : "") << histogram[i];
    std::cout << "],\"selected_levels\":[";
    for (int i = 0; i < frames; ++i) std::cout << (i ? "," : "") << selected[i];
    std::cout << "],\"kinematic_states\":[";
    for(size_t frame=0;frame<kinematicHistory.size();++frame) {
        if(frame) std::cout << ',';
        std::cout << '[';
        for(size_t body=0;body<kinematicHistory[frame].size();++body) {
            if(body) std::cout << ',';
            std::cout << '[';
            for(int k=0;k<6;++k) std::cout << (k?",":"") << kinematicHistory[frame][body][k];
            std::cout << ']';
        }
        std::cout << ']';
    }
    std::cout << "]}\n";
    b2DestroyWorld(world); return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
