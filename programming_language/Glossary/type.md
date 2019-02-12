# [Nominal Typing / Nominal Subtyping](https://en.wikipedia.org/wiki/Nominal_type_system)

**Nominal typing** means that two variables are type-compatible if and only if their declarations name the same type.

**Nominal subtyping** means that one type is a subtype of another if and only if it is explicitly declared to be so in its definition.

## [Abstract Type](https://en.wikipedia.org/wiki/Abstract_type)

1. An abstract type is a type _**in a nominative type system**_ that cannot be instantiated directly.
1. A type that is not abstract – which can be instantiated – is called a concrete type.
1. Every instance of an abstract type is an instance of some concrete subtype. Abstract types are also known as existential types.

# [Boxing](https://en.wikipedia.org/wiki/Object_type_%28object-oriented_programming%29#Boxing)

Boxing is the process of placing a primitive type within an object so that the primitive can be used as a reference object.

* Repeated boxing and unboxing of objects can have a severe performance impact, because boxing dynamically allocates new objects and unboxing (if the boxed value is no longer used) then makes them eligible for garbage collection.
* The boxed object is always a copy of the value object, and is _**usually immutable**_.

## [Autoboxing](https://en.wikipedia.org/wiki/Object_type_%28object-oriented_programming%29#Autoboxing)

Autoboxing is the term for getting a reference type out of a value type just through type conversion (either implicit or explicit). The compiler automatically supplies the extra source code that creates the object.
